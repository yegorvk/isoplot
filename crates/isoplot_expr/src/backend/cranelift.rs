use std::{
    collections::HashMap,
    iter,
    marker::PhantomData,
    mem::{self, MaybeUninit},
    sync::Arc,
};

use cranelift::{
    codegen::ir::FuncRef,
    jit::{JITBuilder, JITModule},
    module::{FuncId, Linkage, Module, default_libcall_names},
    native,
    prelude::*,
};

use crate::{
    MIN_F32_MAGNITUDE,
    artifact::Artifact,
    kernel::{Evaluate, Kernel},
    schema::{FieldType, Schema, WriteSchema},
};
use isoplot_tape::{Instr, Tape};

fn ensure_magnitude(x: f32) -> f32 {
    x.abs().max(MIN_F32_MAGNITUDE).copysign(x)
}

fn ensure_positive(x: f32) -> f32 {
    x.max(MIN_F32_MAGNITUDE)
}

type SingleEvalFunc = extern "C" fn(*const ()) -> f32;
type MultiEvalFunc = extern "C" fn(*const (), *mut ());

pub struct Cranelift<Params, Ret> {
    num_args: usize,
    num_results: usize,
    module: Arc<ModuleGuard>,
    code: *const u8,
    _marker: PhantomData<fn(Params) -> Ret>,
}

// Safety: `code` points into the finalized module, which is kept alive by `module`.
unsafe impl<Params, Ret> Send for Cranelift<Params, Ret> {}
unsafe impl<Params, Ret> Sync for Cranelift<Params, Ret> {}

impl<Params, Ret> Clone for Cranelift<Params, Ret> {
    fn clone(&self) -> Self {
        Self {
            num_args: self.num_args,
            num_results: self.num_results,
            module: Arc::clone(&self.module),
            code: self.code,
            _marker: PhantomData,
        }
    }
}

impl<Params: Schema, Ret: Schema> Cranelift<Params, Ret> {
    pub fn new(artifact: &Artifact<Params, Ret>) -> Self {
        let tape = artifact.tape();
        let (module, code) = compile::<Params, Ret>(&tape, Ret::NUM_FIELDS != 1);

        Self {
            num_args: tape.num_params(),
            num_results: tape.num_results(),
            module: Arc::new(ModuleGuard(Some(module))),
            code,
            _marker: PhantomData,
        }
    }
}

impl<Params: Schema, Ret: Schema> Kernel<Params, Ret> for Cranelift<Params, Ret> {
    type Evaluator<'a>
        = &'a Self
    where
        Self: 'a;

    fn evaluator(&self) -> Self::Evaluator<'_> {
        self
    }
}

unsafe impl<Params: Schema, Ret: Schema> Evaluate<Params, Ret> for Cranelift<Params, Ret> {
    #[inline]
    fn evaluate(&self, params: Params) -> Ret {
        let mut result: MaybeUninit<Ret> = MaybeUninit::uninit();

        if Ret::NUM_FIELDS == 1 {
            let func = unsafe { mem::transmute::<*const u8, SingleEvalFunc>(self.code) };
            let raw_result = func(&params as *const Params as *const ());
            match Ret::field(0).ty {
                FieldType::I32 => result.write_i32(0, [raw_result.to_bits() as i32]),
                FieldType::F32 => result.write_f32(0, [raw_result]),
            };
        } else {
            let func = unsafe { mem::transmute::<*const u8, MultiEvalFunc>(self.code) };
            func(
                &params as *const Params as *const (),
                result.as_mut_ptr().cast(),
            );
        }

        unsafe { result.assume_init() }
    }
}

fn compile<Params: Schema, Ret: Schema>(tape: &Tape, multi: bool) -> (JITModule, *const u8) {
    let mut module = create_module();
    let lib_funcs = declare_lib_funcs(&mut module);

    let mut ctx = module.make_context();
    let func_id = declare_eval(&mut module, &mut ctx.func.signature, multi);

    Translator::<Params, Ret> {
        builder: FunctionBuilder::new(&mut ctx.func, &mut FunctionBuilderContext::new()),
        module: &mut module,
        lib_funcs,
        _marker: PhantomData,
    }
    .translate(tape, multi);

    module.define_function(func_id, &mut ctx).unwrap();
    module.clear_context(&mut ctx);
    module.finalize_definitions().unwrap();
    let code = module.get_finalized_function(func_id);
    (module, code)
}

struct ModuleGuard(Option<JITModule>);

// Safety: the module is finalized and never accessed again except to free it on drop.
unsafe impl Send for ModuleGuard {}
unsafe impl Sync for ModuleGuard {}

impl Drop for ModuleGuard {
    fn drop(&mut self) {
        unsafe {
            self.0.take().unwrap().free_memory();
        }
    }
}

struct LibFuncDesc {
    name: &'static str,
    ptr: *const u8,
    ty: &'static [Type],
}

impl LibFuncDesc {
    unsafe fn new(name: &'static str, ptr: *const u8, ty: &'static [Type]) -> Self {
        Self { name, ptr, ty }
    }

    fn declare(&self, module: &mut JITModule) -> FuncId {
        let mut sig = module.make_signature();
        sig.params
            .extend(self.ty.iter().map(|&ty| AbiParam::new(ty)));
        sig.returns.push(AbiParam::new(types::F32));
        module
            .declare_function(self.name, Linkage::Import, &sig)
            .unwrap()
    }
}

fn default_lib_funcs() -> impl Iterator<Item = LibFuncDesc> {
    let funcs = unsafe {
        [
            LibFuncDesc::new("powf", powf as *const u8, &[types::F32, types::F32]),
            LibFuncDesc::new("powi", powi as *const u8, &[types::F32, types::I32]),
            LibFuncDesc::new("exp", exp as *const u8, &[types::F32]),
            LibFuncDesc::new("ln", ln as *const u8, &[types::F32]),
            LibFuncDesc::new("lg", lg as *const u8, &[types::F32]),
            LibFuncDesc::new("sin", sin as *const u8, &[types::F32]),
            LibFuncDesc::new("cos", cos as *const u8, &[types::F32]),
            LibFuncDesc::new("tan", tan as *const u8, &[types::F32]),
            LibFuncDesc::new("cot", cot as *const u8, &[types::F32]),
        ]
    };
    funcs.into_iter()
}

fn create_module() -> JITModule {
    let mut flags = settings::builder();
    flags.set("use_colocated_libcalls", "false").unwrap();
    flags.set("is_pic", "false").unwrap();
    flags.set("opt_level", "speed").unwrap();

    let isa = native::builder()
        .unwrap()
        .finish(settings::Flags::new(flags))
        .unwrap();

    let mut jit_builder = JITBuilder::with_isa(isa, default_libcall_names());
    for func in default_lib_funcs() {
        jit_builder.symbol(func.name, func.ptr);
    }

    JITModule::new(jit_builder)
}

type LibFuncMap = HashMap<&'static str, (FuncId, Option<FuncRef>)>;

fn declare_lib_funcs(module: &mut JITModule) -> LibFuncMap {
    default_lib_funcs()
        .map(|func| (func.name, (func.declare(module), None)))
        .collect()
}

fn declare_eval(module: &mut JITModule, sig: &mut Signature, multi: bool) -> FuncId {
    let ptr_type = module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type));
    if multi {
        sig.params.push(AbiParam::new(ptr_type));
    } else {
        sig.returns.push(AbiParam::new(types::F32));
    }
    module
        .declare_function("eval", Linkage::Export, sig)
        .unwrap()
}

struct Translator<'a, Params, Ret> {
    builder: FunctionBuilder<'a>,
    module: &'a mut JITModule,
    lib_funcs: LibFuncMap,
    _marker: PhantomData<fn(Params) -> Ret>,
}

impl<Params: Schema, Ret: Schema> Translator<'_, Params, Ret> {
    fn translate(mut self, tape: &Tape, multi: bool) {
        let block = self.builder.create_block();
        self.builder.append_block_params_for_function_params(block);
        self.builder.switch_to_block(block);
        self.builder.seal_block(block);
        let base = self.builder.block_params(block)[0];

        let mut values = Vec::with_capacity(tape.num_params() + tape.instrs().len());
        for field in Params::fields() {
            let offset = i32::try_from(field.offset).unwrap();
            values.push(
                self.builder
                    .ins()
                    .load(types::F32, MemFlagsData::trusted(), base, offset),
            );
        }

        for instr in tape.instrs() {
            let result = self.translate_instr(*instr, &values);
            values.push(result);
        }

        if multi {
            let out = self.builder.block_params(block)[1];
            let results = &values[values.len() - tape.num_results()..];
            for (&result, field) in iter::zip(results.iter(), Ret::fields()) {
                let offset = i32::try_from(field.offset).unwrap();
                self.builder
                    .ins()
                    .store(MemFlagsData::trusted(), result, out, offset);
            }
            self.builder.ins().return_(&[]);
        } else {
            let result = *values.last().unwrap();
            self.builder.ins().return_(&[result]);
        }

        let config = self.module.target_config();
        self.builder.finalize(config);
    }

    fn translate_instr(&mut self, instr: Instr, values: &[Value]) -> Value {
        match instr {
            Instr::I32Const(c) => self.builder.ins().iconst(types::I32, c as i64),
            Instr::BoolConst(c) => self.builder.ins().iconst(types::I8, c as i64),
            Instr::F32Const(c) => self.builder.ins().f32const(c),

            Instr::I32Add(lhs, rhs) => self
                .builder
                .ins()
                .iadd(values[lhs.index()], values[rhs.index()]),
            Instr::I32Sub(lhs, rhs) => self
                .builder
                .ins()
                .isub(values[lhs.index()], values[rhs.index()]),
            Instr::I32Mul(lhs, rhs) => self
                .builder
                .ins()
                .imul(values[lhs.index()], values[rhs.index()]),

            Instr::I32Eq(lhs, rhs) => {
                self.builder
                    .ins()
                    .icmp(IntCC::Equal, values[lhs.index()], values[rhs.index()])
            }
            Instr::I32Ne(lhs, rhs) => {
                self.builder
                    .ins()
                    .icmp(IntCC::NotEqual, values[lhs.index()], values[rhs.index()])
            }
            Instr::I32Lt(lhs, rhs) => self.builder.ins().icmp(
                IntCC::SignedLessThan,
                values[lhs.index()],
                values[rhs.index()],
            ),
            Instr::I32Le(lhs, rhs) => self.builder.ins().icmp(
                IntCC::SignedLessThanOrEqual,
                values[lhs.index()],
                values[rhs.index()],
            ),
            Instr::I32Gt(lhs, rhs) => self.builder.ins().icmp(
                IntCC::SignedGreaterThan,
                values[lhs.index()],
                values[rhs.index()],
            ),
            Instr::I32Ge(lhs, rhs) => self.builder.ins().icmp(
                IntCC::SignedGreaterThanOrEqual,
                values[lhs.index()],
                values[rhs.index()],
            ),

            Instr::Not(src) => self
                .builder
                .ins()
                .icmp_imm_u(IntCC::Equal, values[src.index()], 0),
            Instr::And(lhs, rhs) => self
                .builder
                .ins()
                .band(values[lhs.index()], values[rhs.index()]),
            Instr::Or(lhs, rhs) => self
                .builder
                .ins()
                .bor(values[lhs.index()], values[rhs.index()]),
            Instr::Xor(lhs, rhs) => self
                .builder
                .ins()
                .bxor(values[lhs.index()], values[rhs.index()]),

            Instr::I32FromBool(src) => self.builder.ins().uextend(types::I32, values[src.index()]),
            Instr::F32FromI32(src) => self
                .builder
                .ins()
                .fcvt_from_sint(types::F32, values[src.index()]),
            Instr::F32FromBool(src) => {
                let int = self.builder.ins().uextend(types::I32, values[src.index()]);
                self.builder.ins().fcvt_from_uint(types::F32, int)
            }

            Instr::CopyI32(src) => values[src.index()],
            Instr::CopyBool(src) => values[src.index()],
            Instr::CopyF32(src) => values[src.index()],

            Instr::F32Neg(src) => self.builder.ins().fneg(values[src.index()]),
            Instr::F32Abs(src) => self.builder.ins().fabs(values[src.index()]),
            Instr::F32Sign(src) => {
                let one = self.builder.ins().f32const(1.0);
                self.builder.ins().fcopysign(one, values[src.index()])
            }
            Instr::F32Floor(src) => self.builder.ins().floor(values[src.index()]),
            Instr::F32Add(lhs, rhs) => self
                .builder
                .ins()
                .fadd(values[lhs.index()], values[rhs.index()]),
            Instr::F32Sub(lhs, rhs) => self
                .builder
                .ins()
                .fsub(values[lhs.index()], values[rhs.index()]),
            Instr::F32Mul(lhs, rhs) => self
                .builder
                .ins()
                .fmul(values[lhs.index()], values[rhs.index()]),
            Instr::F32Div(lhs, rhs) => {
                let min = self.builder.ins().f32const(MIN_F32_MAGNITUDE);
                let magnitude = self.builder.ins().fabs(values[rhs.index()]);
                let clamped = self.builder.ins().fmax(magnitude, min);
                let divisor = self.builder.ins().fcopysign(clamped, values[rhs.index()]);
                self.builder.ins().fdiv(values[lhs.index()], divisor)
            }

            Instr::F32Min(lhs, rhs) => self
                .builder
                .ins()
                .fmin(values[lhs.index()], values[rhs.index()]),
            Instr::F32Max(lhs, rhs) => self
                .builder
                .ins()
                .fmax(values[lhs.index()], values[rhs.index()]),

            Instr::F32Powf(lhs, rhs) => {
                self.call("powf", &[values[lhs.index()], values[rhs.index()]])
            }
            Instr::F32Powi(lhs, rhs) => {
                self.call("powi", &[values[lhs.index()], values[rhs.index()]])
            }

            Instr::F32Exp(src) => self.call("exp", &[values[src.index()]]),
            Instr::F32Ln(src) => self.call("ln", &[values[src.index()]]),
            Instr::F32Lg(src) => self.call("lg", &[values[src.index()]]),
            Instr::F32Sin(src) => self.call("sin", &[values[src.index()]]),
            Instr::F32Cos(src) => self.call("cos", &[values[src.index()]]),
            Instr::F32Tan(src) => self.call("tan", &[values[src.index()]]),
            Instr::F32Cot(src) => self.call("cot", &[values[src.index()]]),

            Instr::F32Eq(lhs, rhs) => {
                self.builder
                    .ins()
                    .fcmp(FloatCC::Equal, values[lhs.index()], values[rhs.index()])
            }
            Instr::F32Ne(lhs, rhs) => {
                self.builder
                    .ins()
                    .fcmp(FloatCC::NotEqual, values[lhs.index()], values[rhs.index()])
            }
            Instr::F32Lt(lhs, rhs) => {
                self.builder
                    .ins()
                    .fcmp(FloatCC::LessThan, values[lhs.index()], values[rhs.index()])
            }
            Instr::F32Le(lhs, rhs) => self.builder.ins().fcmp(
                FloatCC::LessThanOrEqual,
                values[lhs.index()],
                values[rhs.index()],
            ),
            Instr::F32Gt(lhs, rhs) => self.builder.ins().fcmp(
                FloatCC::GreaterThan,
                values[lhs.index()],
                values[rhs.index()],
            ),
            Instr::F32Ge(lhs, rhs) => self.builder.ins().fcmp(
                FloatCC::GreaterThanOrEqual,
                values[lhs.index()],
                values[rhs.index()],
            ),

            Instr::I32Sel(cond, v_true, v_false) => self.builder.ins().select(
                values[cond.index()],
                values[v_true.index()],
                values[v_false.index()],
            ),
            Instr::F32Sel(cond, v_true, v_false) => self.builder.ins().select(
                values[cond.index()],
                values[v_true.index()],
                values[v_false.index()],
            ),
        }
    }

    fn call(&mut self, name: &'static str, args: &[Value]) -> Value {
        let (id, func_ref) = self.lib_funcs.get_mut(name).unwrap();
        let func_ref = *func_ref
            .get_or_insert_with(|| self.module.declare_func_in_func(*id, self.builder.func));

        let inst = self.builder.ins().call(func_ref, args);
        self.builder.inst_results(inst)[0]
    }
}

extern "C" fn powf(x: f32, y: f32) -> f32 {
    ensure_magnitude(x).powf(y)
}

extern "C" fn powi(x: f32, n: i32) -> f32 {
    ensure_magnitude(x).powi(n)
}

extern "C" fn exp(x: f32) -> f32 {
    x.exp()
}

extern "C" fn ln(x: f32) -> f32 {
    ensure_positive(x).ln()
}

extern "C" fn lg(x: f32) -> f32 {
    ensure_positive(x).log10()
}

extern "C" fn sin(x: f32) -> f32 {
    x.sin()
}

extern "C" fn cos(x: f32) -> f32 {
    x.cos()
}

extern "C" fn tan(x: f32) -> f32 {
    x.tan()
}

extern "C" fn cot(x: f32) -> f32 {
    x.tan().recip()
}

#[cfg(test)]
mod tests {
    use super::*;
    use isoplot_tape::ValueType;

    #[test]
    fn f32_ops() {
        let mut b = Tape::builder(vec![ValueType::F32, ValueType::F32], vec![ValueType::F32]);
        let [x, y] = [b.param(0), b.param(1)];

        let sum = b.f32_add(x, y);
        let prod = b.f32_mul(sum, x);
        let diff = b.f32_sub(prod, y);
        let quot = b.f32_div(diff, y);
        let neg = b.f32_neg(quot);
        let abs = b.f32_abs(neg);
        let abs = b.copy_f32(abs);
        let min = b.f32_min(abs, x);
        let max = b.f32_max(min, y);
        let c = b.f32_const(0.5);
        b.f32_add(max, c);

        let artifact: Artifact<_, f32> = Artifact::from_tape(b.build().unwrap());

        let (v_x, v_y) = (1.75f32, -0.5f32);
        let expected = (-((v_x + v_y) * v_x - v_y) / v_y).abs().min(v_x).max(v_y) + 0.5;

        assert_eq!(
            Cranelift::new(&artifact).evaluator().evaluate([v_x, v_y]),
            expected
        );
    }

    #[test]
    fn f32_fn_calls() {
        let mut b = Tape::builder(vec![ValueType::F32, ValueType::F32], vec![ValueType::F32]);
        let x = b.param(0);
        let y = b.param(1);
        let sin = b.f32_sin(x);
        let cos = b.f32_cos(y);
        let tan = b.f32_tan(x);
        let cot = b.f32_cot(y);
        let exp = b.f32_exp(x);
        let ln = b.f32_ln(y);
        let lg = b.f32_lg(x);
        let pow = b.f32_powf(x, y);

        let mut acc = sin;
        for v in [cos, tan, cot, exp, ln, lg, pow] {
            acc = b.f32_add(acc, v);
        }

        let artifact: Artifact<_, f32> = Artifact::from_tape(b.build().unwrap());

        let (x, y) = (0.7f32, 1.3f32);
        let expected = x.sin()
            + y.cos()
            + x.tan()
            + y.tan().recip()
            + x.exp()
            + y.ln()
            + x.log10()
            + x.powf(y);

        assert_eq!(
            Cranelift::new(&artifact).evaluator().evaluate([x, y]),
            expected
        );
    }

    #[test]
    fn i32_ops() {
        let mut b = Tape::builder(vec![ValueType::F32], vec![ValueType::F32]);
        let x = b.param(0);
        let three = b.i32_const(3);
        let two = b.i32_const(2);
        let five = b.i32_add(three, two);
        let diff = b.i32_sub(five, two);
        let six = b.i32_mul(diff, two);
        let sixf = b.f32_from_i32(six);
        let cube = b.f32_powi(x, diff);
        b.f32_add(sixf, cube);

        let artifact: Artifact<_, f32> = Artifact::from_tape(b.build().unwrap());

        let x = 1.5f32;
        assert_eq!(
            Cranelift::new(&artifact).evaluator().evaluate(x),
            6.0 + x.powi(3)
        );
    }

    #[test]
    fn f32_div_clamped() {
        let mut b = Tape::builder(vec![ValueType::F32, ValueType::F32], vec![ValueType::F32]);
        let x = b.param(0);
        let y = b.param(1);
        b.f32_div(x, y);

        let artifact: Artifact<_, f32> = Artifact::from_tape(b.build().unwrap());

        let kernel = Cranelift::new(&artifact);
        let evaluator = kernel.evaluator();

        let peak = 1.0 / MIN_F32_MAGNITUDE;
        assert_eq!(evaluator.evaluate([3.0, -2.0]), -1.5);
        assert_eq!(evaluator.evaluate([1.0, 0.0]), peak);
        assert_eq!(evaluator.evaluate([1.0, -0.0]), -peak);
        assert_eq!(evaluator.evaluate([2.0, 1.0e-40]), 2.0 * peak);
    }

    #[test]
    fn f32_sign() {
        let mut b = Tape::builder(vec![ValueType::F32], vec![ValueType::F32]);
        let x = b.param(0);
        b.f32_sign(x);

        let artifact: Artifact<_, f32> = Artifact::from_tape(b.build().unwrap());

        let kernel = Cranelift::new(&artifact);
        let evaluator = kernel.evaluator();

        assert_eq!(evaluator.evaluate(-3.5), -1.0);
        assert_eq!(evaluator.evaluate(2.0), 1.0);
        assert_eq!(evaluator.evaluate(0.0), 1.0);
        assert_eq!(evaluator.evaluate(-0.0), -1.0);
    }

    #[test]
    fn f32_select() {
        // f = if 0 < x && x < y { x } else { y }
        let mut b = Tape::builder(vec![ValueType::F32, ValueType::F32], vec![ValueType::F32]);
        let x = b.param(0);
        let y = b.param(1);
        let zero = b.f32_const(0.0);
        let lt = b.f32_lt(x, y);
        let pos = b.f32_gt(x, zero);
        let both = b.and(lt, pos);
        b.f32_sel(both, x, y);

        let artifact: Artifact<_, f32> = Artifact::from_tape(b.build().unwrap());

        let kernel = Cranelift::new(&artifact);
        let evaluator = kernel.evaluator();

        assert_eq!(evaluator.evaluate([1.0, 2.0]), 1.0);
        assert_eq!(evaluator.evaluate([-1.0, 2.0]), 2.0);
        assert_eq!(evaluator.evaluate([3.0, 2.0]), 2.0);
    }

    #[test]
    fn bool_i32_ops() {
        let mut b = Tape::builder(vec![ValueType::F32], vec![ValueType::F32]);
        let x = b.param(0);
        let two = b.i32_const(2);
        let three = b.i32_const(3);
        let lt = b.i32_lt(two, three); // true
        let ge = b.i32_ge(two, three); // false
        let yes = b.bool_const(true);
        let xor = b.xor(lt, yes); // false
        let any = b.or(xor, ge); // false
        let not = b.not(any); // true
        let sel = b.i32_sel(not, two, three); // 2
        let bump = b.i32_from_bool(lt); // 1
        let sum = b.i32_add(sel, bump); // 3
        let sum = b.f32_from_i32(sum);
        let one = b.f32_from_bool(not); // 1.0
        let sum = b.f32_add(sum, one); // 4.0
        b.f32_add(sum, x);

        let artifact: Artifact<_, f32> = Artifact::from_tape(b.build().unwrap());
        assert_eq!(Cranelift::new(&artifact).evaluator().evaluate(0.5), 4.5);
    }

    #[test]
    fn multi_results() {
        let mut b = Tape::builder(
            vec![ValueType::F32, ValueType::F32],
            vec![ValueType::F32; 3],
        );
        let x = b.param(0);
        let y = b.param(1);
        b.f32_add(x, y);
        b.f32_mul(x, y);
        b.f32_sub(x, y);

        let artifact: Artifact<_, [f32; 3]> = Artifact::from_tape(b.build().unwrap());

        let (x, y) = (1.5f32, 2.25f32);
        assert_eq!(
            Cranelift::new(&artifact).evaluator().evaluate([x, y]),
            [x + y, x * y, x - y]
        );
    }
}
