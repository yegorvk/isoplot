use std::{collections::HashMap, marker::PhantomData, mem, sync::Arc};

use cranelift::codegen::ir::{
    AbiParam, FuncRef, InstBuilder, MemFlagsData, Signature, Type, Value, types,
};
use cranelift::codegen::settings::{self, Configurable};
use cranelift::frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift::jit::{JITBuilder, JITModule};
use cranelift::module::{FuncId, Linkage, Module, default_libcall_names};
use cranelift::native;

use crate::{
    layout::{RawValue, Vector},
    tape::{Instr, Tape, ValueId},
};

type ScalarEvalFunc = extern "C" fn(*const f32) -> f32;
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
            let func = unsafe { mem::transmute::<*const u8, ScalarEvalFunc>(self.code) };
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

        for &instr in tape.instrs() {
            let result = self.translate_instr(instr, &values);
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
        let v = |id: ValueId| values[id.index()];

        match instr {
            Instr::I32Const(c) => self.builder.ins().iconst(types::I32, c as i64),
            Instr::F32Const(c) => self.builder.ins().f32const(c),

            Instr::I32Add(lhs, rhs) => self.builder.ins().iadd(v(lhs), v(rhs)),
            Instr::I32Sub(lhs, rhs) => self.builder.ins().isub(v(lhs), v(rhs)),
            Instr::I32Mul(lhs, rhs) => self.builder.ins().imul(v(lhs), v(rhs)),

            Instr::F32FromI32(src) => self.builder.ins().fcvt_from_sint(types::F32, v(src)),

            Instr::Copy(src) => v(src),

            Instr::F32Neg(src) => self.builder.ins().fneg(v(src)),
            Instr::F32Abs(src) => self.builder.ins().fabs(v(src)),
            Instr::F32Sign(src) => {
                let one = self.builder.ins().f32const(1.0);
                self.builder.ins().fcopysign(one, v(src))
            }
            Instr::F32Floor(src) => self.builder.ins().floor(v(src)),
            Instr::F32Add(lhs, rhs) => self.builder.ins().fadd(v(lhs), v(rhs)),
            Instr::F32Sub(lhs, rhs) => self.builder.ins().fsub(v(lhs), v(rhs)),
            Instr::F32Mul(lhs, rhs) => self.builder.ins().fmul(v(lhs), v(rhs)),
            Instr::F32Div(lhs, rhs) => self.builder.ins().fdiv(v(lhs), v(rhs)),

            Instr::F32Min(lhs, rhs) => self.builder.ins().fmin(v(lhs), v(rhs)),
            Instr::F32Max(lhs, rhs) => self.builder.ins().fmax(v(lhs), v(rhs)),

            Instr::F32Powf(lhs, rhs) => self.call("powf", &[v(lhs), v(rhs)]),
            Instr::F32Powi(lhs, rhs) => self.call("powi", &[v(lhs), v(rhs)]),

            Instr::F32Exp(src) => self.call("exp", &[v(src)]),
            Instr::F32Ln(src) => self.call("ln", &[v(src)]),
            Instr::F32Lg(src) => self.call("lg", &[v(src)]),
            Instr::F32Sin(src) => self.call("sin", &[v(src)]),
            Instr::F32Cos(src) => self.call("cos", &[v(src)]),
            Instr::F32Tan(src) => self.call("tan", &[v(src)]),
            Instr::F32Cot(src) => self.call("cot", &[v(src)]),
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
        let sum = b.instr(Instr::F32Add(x, y));
        let prod = b.instr(Instr::F32Mul(sum, x));
        let diff = b.instr(Instr::F32Sub(prod, y));
        let quot = b.instr(Instr::F32Div(diff, y));
        let neg = b.instr(Instr::F32Neg(quot));
        let abs = b.instr(Instr::F32Abs(neg));
        let abs = b.instr(Instr::Copy(abs));
        let min = b.instr(Instr::F32Min(abs, x));
        let max = b.instr(Instr::F32Max(min, y));
        let c = b.instr(Instr::F32Const(0.5));
        b.instr(Instr::F32Add(max, c));
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
        let sin = b.instr(Instr::F32Sin(x));
        let cos = b.instr(Instr::F32Cos(y));
        let tan = b.instr(Instr::F32Tan(x));
        let cot = b.instr(Instr::F32Cot(y));
        let exp = b.instr(Instr::F32Exp(x));
        let ln = b.instr(Instr::F32Ln(y));
        let lg = b.instr(Instr::F32Lg(x));
        let pow = b.instr(Instr::F32Powf(x, y));

        let mut acc = sin;
        for v in [cos, tan, cot, exp, ln, lg, pow] {
            acc = b.instr(Instr::F32Add(acc, v));
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
        let three = b.instr(Instr::I32Const(3));
        let two = b.instr(Instr::I32Const(2));
        let five = b.instr(Instr::I32Add(three, two));
        let diff = b.instr(Instr::I32Sub(five, two));
        let six = b.instr(Instr::I32Mul(diff, two));
        let sixf = b.instr(Instr::F32FromI32(six));
        let cube = b.instr(Instr::F32Powi(x, diff));
        b.instr(Instr::F32Add(sixf, cube));

        let tape = b.build().unwrap();
        let inst = Instance::<f32>::new(&tape);

        let x = 1.5f32;
        assert_eq!(eval(&inst, &[x]), 6.0 + x.powi(3));
    }

    #[test]
    fn f32_sign() {
        let mut b = Tape::builder(vec![Type::F32], vec![Type::F32]);
        let x = b.arg(0);
        b.instr(Instr::F32Sign(x));
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
        b.instr(Instr::F32Floor(x));
        let tape = b.build().unwrap();
        let inst = Instance::<f32>::new(&tape);

        assert_eq!(eval(&inst, &[2.7]), 2.0);
        assert_eq!(eval(&inst, &[-2.3]), -3.0);
        assert_eq!(eval(&inst, &[4.0]), 4.0);
    }

    #[test]
    fn multi_results() {
        let mut b = Tape::builder(vec![Type::F32, Type::F32], vec![Type::F32; 3]);
        let x = b.arg(0);
        let y = b.arg(1);
        b.instr(Instr::F32Add(x, y));
        b.instr(Instr::F32Mul(x, y));
        b.instr(Instr::F32Sub(x, y));
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
        let c = b.instr(Instr::F32Const(1.0));
        b.instr(Instr::F32Add(x, c));

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
