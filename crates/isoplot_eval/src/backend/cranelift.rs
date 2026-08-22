use std::{collections::HashMap, marker::PhantomData, mem, ops::Index, sync::Arc};

use cranelift::codegen::ir::{
    AbiParam, FuncRef, InstBuilder, MemFlagsData, Signature, Type, Value,
    condcodes::{FloatCC, IntCC},
    types,
};
use cranelift::codegen::settings::{self, Configurable};
use cranelift::frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift::jit::{JITBuilder, JITModule};
use cranelift::module::{FuncId, Linkage, Module, default_libcall_names};
use cranelift::native;

use crate::{
    layout::{RawValue, Vector},
    tape::{Instr, Tape, ValueId, ValuePrimitive},
};

type SingleEvalFunc = extern "C" fn(*const f32) -> f32;
type MultiEvalFunc = extern "C" fn(*const f32, *mut f32);

pub(super) struct Instance<Ret> {
    num_args: usize,
    num_results: usize,
    module: Arc<ModuleGuard>,
    code: *const u8,
    _marker: PhantomData<fn() -> Ret>,
}

// Safety: `code` points into the finalized module, which is kept alive by `module`.
unsafe impl<Ret> Send for Instance<Ret> {}
unsafe impl<Ret> Sync for Instance<Ret> {}

impl<Ret> Clone for Instance<Ret> {
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

impl<Ret: Vector> Instance<Ret> {
    pub(super) fn new(tape: &Tape) -> Self {
        let (module, code) = compile(tape, Ret::LEN != 1);

        Self {
            num_args: tape.num_args(),
            num_results: tape.num_results(),
            module: Arc::new(ModuleGuard(Some(module))),
            code,
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    pub(super) fn evaluate_into(&self, args: &[RawValue], results: &mut [RawValue]) {
        debug_assert_eq!(args.len(), self.num_args);
        debug_assert_eq!(results.len(), self.num_results);

        if Ret::LEN == 1 {
            let func = unsafe { mem::transmute::<*const u8, SingleEvalFunc>(self.code) };
            results[0] = RawValue::from_f32(func(args.as_ptr().cast()));
        } else {
            let func = unsafe { mem::transmute::<*const u8, MultiEvalFunc>(self.code) };
            func(args.as_ptr().cast(), results.as_mut_ptr().cast());
        }
    }
}

fn compile(tape: &Tape, multi: bool) -> (JITModule, *const u8) {
    let mut module = create_module();
    let lib_funcs = declare_lib_funcs(&mut module);

    let mut ctx = module.make_context();
    let func_id = declare_eval(&mut module, &mut ctx.func.signature, multi);

    Translator {
        builder: FunctionBuilder::new(&mut ctx.func, &mut FunctionBuilderContext::new()),
        module: &mut module,
        lib_funcs,
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

impl<T: ValuePrimitive> Index<ValueId<T>> for [Value] {
    type Output = Value;

    fn index(&self, id: ValueId<T>) -> &Value {
        &self[id.index()]
    }
}

struct Translator<'a> {
    builder: FunctionBuilder<'a>,
    module: &'a mut JITModule,
    lib_funcs: LibFuncMap,
}

impl Translator<'_> {
    fn translate(mut self, tape: &Tape, multi: bool) {
        let block = self.builder.create_block();
        self.builder.append_block_params_for_function_params(block);
        self.builder.switch_to_block(block);
        self.builder.seal_block(block);
        let base = self.builder.block_params(block)[0];

        let mut values = Vec::with_capacity(tape.num_args() + tape.instrs().len());
        for i in 0..tape.num_args() {
            let offset = i32::try_from(4 * i).unwrap();
            values.push(
                self.builder
                    .ins()
                    .load(types::F32, MemFlagsData::trusted(), base, offset),
            );
        }

        for r in tape.instrs() {
            let result = self.translate_instr(r.instr(), &values);
            values.push(result);
        }

        if multi {
            let out = self.builder.block_params(block)[1];
            let results = &values[values.len() - tape.num_results()..];
            for (i, &result) in results.iter().enumerate() {
                let offset = i32::try_from(4 * i).unwrap();
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

            Instr::I32Add(lhs, rhs) => self.builder.ins().iadd(values[lhs], values[rhs]),
            Instr::I32Sub(lhs, rhs) => self.builder.ins().isub(values[lhs], values[rhs]),
            Instr::I32Mul(lhs, rhs) => self.builder.ins().imul(values[lhs], values[rhs]),

            Instr::I32Eq(lhs, rhs) => {
                self.builder
                    .ins()
                    .icmp(IntCC::Equal, values[lhs], values[rhs])
            }
            Instr::I32Ne(lhs, rhs) => {
                self.builder
                    .ins()
                    .icmp(IntCC::NotEqual, values[lhs], values[rhs])
            }
            Instr::I32Lt(lhs, rhs) => {
                self.builder
                    .ins()
                    .icmp(IntCC::SignedLessThan, values[lhs], values[rhs])
            }
            Instr::I32Le(lhs, rhs) => {
                self.builder
                    .ins()
                    .icmp(IntCC::SignedLessThanOrEqual, values[lhs], values[rhs])
            }
            Instr::I32Gt(lhs, rhs) => {
                self.builder
                    .ins()
                    .icmp(IntCC::SignedGreaterThan, values[lhs], values[rhs])
            }
            Instr::I32Ge(lhs, rhs) => {
                self.builder
                    .ins()
                    .icmp(IntCC::SignedGreaterThanOrEqual, values[lhs], values[rhs])
            }

            Instr::Not(src) => self.builder.ins().icmp_imm_u(IntCC::Equal, values[src], 0),
            Instr::And(lhs, rhs) => self.builder.ins().band(values[lhs], values[rhs]),
            Instr::Or(lhs, rhs) => self.builder.ins().bor(values[lhs], values[rhs]),
            Instr::Xor(lhs, rhs) => self.builder.ins().bxor(values[lhs], values[rhs]),

            Instr::I32FromBool(src) => self.builder.ins().uextend(types::I32, values[src]),
            Instr::F32FromI32(src) => self.builder.ins().fcvt_from_sint(types::F32, values[src]),
            Instr::F32FromBool(src) => {
                let int = self.builder.ins().uextend(types::I32, values[src]);
                self.builder.ins().fcvt_from_uint(types::F32, int)
            }

            Instr::CopyI32(src) => values[src],
            Instr::CopyBool(src) => values[src],
            Instr::CopyF32(src) => values[src],

            Instr::F32Neg(src) => self.builder.ins().fneg(values[src]),
            Instr::F32Abs(src) => self.builder.ins().fabs(values[src]),
            Instr::F32Sign(src) => {
                let one = self.builder.ins().f32const(1.0);
                self.builder.ins().fcopysign(one, values[src])
            }
            Instr::F32Floor(src) => self.builder.ins().floor(values[src]),
            Instr::F32Add(lhs, rhs) => self.builder.ins().fadd(values[lhs], values[rhs]),
            Instr::F32Sub(lhs, rhs) => self.builder.ins().fsub(values[lhs], values[rhs]),
            Instr::F32Mul(lhs, rhs) => self.builder.ins().fmul(values[lhs], values[rhs]),
            Instr::F32Div(lhs, rhs) => self.builder.ins().fdiv(values[lhs], values[rhs]),

            Instr::F32Min(lhs, rhs) => self.builder.ins().fmin(values[lhs], values[rhs]),
            Instr::F32Max(lhs, rhs) => self.builder.ins().fmax(values[lhs], values[rhs]),

            Instr::F32Powf(lhs, rhs) => self.call("powf", &[values[lhs], values[rhs]]),
            Instr::F32Powi(lhs, rhs) => self.call("powi", &[values[lhs], values[rhs]]),

            Instr::F32Exp(src) => self.call("exp", &[values[src]]),
            Instr::F32Ln(src) => self.call("ln", &[values[src]]),
            Instr::F32Lg(src) => self.call("lg", &[values[src]]),
            Instr::F32Sin(src) => self.call("sin", &[values[src]]),
            Instr::F32Cos(src) => self.call("cos", &[values[src]]),
            Instr::F32Tan(src) => self.call("tan", &[values[src]]),
            Instr::F32Cot(src) => self.call("cot", &[values[src]]),

            Instr::F32Eq(lhs, rhs) => {
                self.builder
                    .ins()
                    .fcmp(FloatCC::Equal, values[lhs], values[rhs])
            }
            Instr::F32Ne(lhs, rhs) => {
                self.builder
                    .ins()
                    .fcmp(FloatCC::NotEqual, values[lhs], values[rhs])
            }
            Instr::F32Lt(lhs, rhs) => {
                self.builder
                    .ins()
                    .fcmp(FloatCC::LessThan, values[lhs], values[rhs])
            }
            Instr::F32Le(lhs, rhs) => {
                self.builder
                    .ins()
                    .fcmp(FloatCC::LessThanOrEqual, values[lhs], values[rhs])
            }
            Instr::F32Gt(lhs, rhs) => {
                self.builder
                    .ins()
                    .fcmp(FloatCC::GreaterThan, values[lhs], values[rhs])
            }
            Instr::F32Ge(lhs, rhs) => {
                self.builder
                    .ins()
                    .fcmp(FloatCC::GreaterThanOrEqual, values[lhs], values[rhs])
            }

            Instr::I32Sel(cond, v_true, v_false) => {
                self.builder
                    .ins()
                    .select(values[cond], values[v_true], values[v_false])
            }
            Instr::F32Sel(cond, v_true, v_false) => {
                self.builder
                    .ins()
                    .select(values[cond], values[v_true], values[v_false])
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::Type;

    fn eval(inst: &Instance<f32>, args: &[f32]) -> f32 {
        let mut output = [RawValue::ZERO];
        inst.evaluate_into(bytemuck::cast_slice(args), &mut output);
        output[0].as_f32()
    }

    #[test]
    fn f32_ops() {
        let mut b = Tape::builder(vec![Type::F32, Type::F32], vec![Type::F32]);
        let x = b.arg(0);
        let y = b.arg(1);
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
        let tape = b.build().unwrap();

        let (x, y) = (1.75f32, -0.5f32);
        let expected = (-((x + y) * x - y) / y).abs().min(x).max(y) + 0.5;
        assert_eq!(eval(&Instance::<f32>::new(&tape), &[x, y]), expected);
    }

    #[test]
    fn f32_fn_calls() {
        let mut b = Tape::builder(vec![Type::F32, Type::F32], vec![Type::F32]);
        let x = b.arg(0);
        let y = b.arg(1);
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

        let tape = b.build().unwrap();
        let inst = Instance::<f32>::new(&tape);

        let (x, y) = (0.7f32, 1.3f32);
        let expected = x.sin()
            + y.cos()
            + x.tan()
            + y.tan().recip()
            + x.exp()
            + y.ln()
            + x.log10()
            + x.powf(y);

        assert_eq!(eval(&inst, &[x, y]), expected);
    }

    #[test]
    fn i32_ops() {
        let mut b = Tape::builder(vec![Type::F32], vec![Type::F32]);
        let x = b.arg(0);
        let three = b.i32_const(3);
        let two = b.i32_const(2);
        let five = b.i32_add(three, two);
        let diff = b.i32_sub(five, two);
        let six = b.i32_mul(diff, two);
        let sixf = b.f32_from_i32(six);
        let cube = b.f32_powi(x, diff);
        b.f32_add(sixf, cube);

        let tape = b.build().unwrap();
        let inst = Instance::<f32>::new(&tape);

        let x = 1.5f32;
        assert_eq!(eval(&inst, &[x]), 6.0 + x.powi(3));
    }

    #[test]
    fn f32_sign() {
        let mut b = Tape::builder(vec![Type::F32], vec![Type::F32]);
        let x = b.arg(0);
        b.f32_sign(x);
        let tape = b.build().unwrap();
        let inst = Instance::<f32>::new(&tape);

        assert_eq!(eval(&inst, &[-3.5]), -1.0);
        assert_eq!(eval(&inst, &[2.0]), 1.0);
        assert_eq!(eval(&inst, &[0.0]), 1.0);
        assert_eq!(eval(&inst, &[-0.0]), -1.0);
    }

    #[test]
    fn f32_floor() {
        let mut b = Tape::builder(vec![Type::F32], vec![Type::F32]);
        let x = b.arg(0);
        b.f32_floor(x);
        let tape = b.build().unwrap();
        let inst = Instance::<f32>::new(&tape);

        assert_eq!(eval(&inst, &[2.7]), 2.0);
        assert_eq!(eval(&inst, &[-2.3]), -3.0);
        assert_eq!(eval(&inst, &[4.0]), 4.0);
    }

    #[test]
    fn f32_select() {
        // f = if 0 < x && x < y { x } else { y }
        let mut b = Tape::builder(vec![Type::F32, Type::F32], vec![Type::F32]);
        let x = b.arg(0);
        let y = b.arg(1);
        let zero = b.f32_const(0.0);
        let lt = b.f32_lt(x, y);
        let pos = b.f32_gt(x, zero);
        let both = b.and(lt, pos);
        b.f32_sel(both, x, y);
        let tape = b.build().unwrap();
        let inst = Instance::<f32>::new(&tape);

        assert_eq!(eval(&inst, &[1.0, 2.0]), 1.0);
        assert_eq!(eval(&inst, &[-1.0, 2.0]), 2.0);
        assert_eq!(eval(&inst, &[3.0, 2.0]), 2.0);
    }

    #[test]
    fn bool_i32_ops() {
        let mut b = Tape::builder(vec![Type::F32], vec![Type::F32]);
        let x = b.arg(0);
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
        let tape = b.build().unwrap();
        let inst = Instance::<f32>::new(&tape);

        assert_eq!(eval(&inst, &[0.5]), 4.5);
    }

    #[test]
    fn multi_results() {
        let mut b = Tape::builder(vec![Type::F32, Type::F32], vec![Type::F32; 3]);
        let x = b.arg(0);
        let y = b.arg(1);
        b.f32_add(x, y);
        b.f32_mul(x, y);
        b.f32_sub(x, y);
        let tape = b.build().unwrap();

        let (x, y) = (1.5f32, 2.25f32);
        let mut results = [0.0f32; 3];
        Instance::<[f32; 3]>::new(&tape).evaluate_into(
            bytemuck::cast_slice(&[x, y]),
            bytemuck::cast_slice_mut(&mut results),
        );
        assert_eq!(results, [x + y, x * y, x - y]);
    }

    #[test]
    fn clone_outlives_original() {
        let mut b = Tape::builder(vec![Type::F32], vec![Type::F32]);
        let x = b.arg(0);
        let c = b.f32_const(1.0);
        b.f32_add(x, c);

        let tape = b.build().unwrap();
        let inst = Instance::<f32>::new(&tape);
        let clone = inst.clone();
        drop(inst);

        assert_eq!(eval(&clone, &[2.0]), 3.0);
    }
}

extern "C" fn powf(x: f32, y: f32) -> f32 {
    x.powf(y)
}

extern "C" fn powi(x: f32, n: i32) -> f32 {
    x.powi(n)
}

extern "C" fn exp(x: f32) -> f32 {
    x.exp()
}

extern "C" fn ln(x: f32) -> f32 {
    x.ln()
}

extern "C" fn lg(x: f32) -> f32 {
    x.log10()
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
