use std::collections::HashMap;
use std::mem;
use std::sync::Arc;

use cranelift::codegen::ir::{
    AbiParam, FuncRef, InstBuilder, MemFlagsData, Signature, Type, Value, types,
};
use cranelift::frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift::jit::{JITBuilder, JITModule};
use cranelift::module::{FuncId, Linkage, Module, default_libcall_names};

use crate::tape::{Instr, Tape, ValueId};

type EvalFunc = extern "C" fn(*const f32) -> f32;

pub(super) struct Instance {
    num_inputs: usize,
    module: Arc<ModuleGuard>,
    func: EvalFunc,
}

impl Clone for Instance {
    fn clone(&self) -> Self {
        Self {
            num_inputs: self.num_inputs,
            module: Arc::clone(&self.module),
            func: self.func,
        }
    }
}

impl Instance {
    pub(super) fn new(tape: &Tape) -> Self {
        let mut module = create_module();
        let lib_funcs = declare_lib_funcs(&mut module);

        let mut ctx = module.make_context();
        let func_id = declare_eval(&mut module, &mut ctx.func.signature);

        Translator {
            b: FunctionBuilder::new(&mut ctx.func, &mut FunctionBuilderContext::new()),
            module: &mut module,
            lib_funcs,
        }
        .translate(tape);

        module.define_function(func_id, &mut ctx).unwrap();
        module.clear_context(&mut ctx);
        module.finalize_definitions().unwrap();
        let code = module.get_finalized_function(func_id);

        Self {
            num_inputs: tape.num_inputs(),
            module: Arc::new(ModuleGuard(Some(module))),
            func: unsafe { mem::transmute::<*const u8, EvalFunc>(code) },
        }
    }

    #[inline(always)]
    pub(super) fn evaluate(&self, inputs: &[f32]) -> f32 {
        assert_eq!(inputs.len(), self.num_inputs);
        (self.func)(inputs.as_ptr())
    }
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
    let mut jit_builder = JITBuilder::new(default_libcall_names()).unwrap();
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

fn declare_eval(module: &mut JITModule, sig: &mut Signature) -> FuncId {
    let ptr_type = module.target_config().pointer_type();
    sig.params.push(AbiParam::new(ptr_type));
    sig.returns.push(AbiParam::new(types::F32));
    module
        .declare_function("eval", Linkage::Export, sig)
        .unwrap()
}

struct Translator<'a> {
    b: FunctionBuilder<'a>,
    module: &'a mut JITModule,
    lib_funcs: LibFuncMap,
}

impl Translator<'_> {
    fn translate(mut self, tape: &Tape) {
        let block = self.b.create_block();
        self.b.append_block_params_for_function_params(block);
        self.b.switch_to_block(block);
        self.b.seal_block(block);
        let base = self.b.block_params(block)[0];

        let mut values = Vec::with_capacity(tape.num_inputs() + tape.instrs().len());
        for i in 0..tape.num_inputs() {
            let offset = i32::try_from(4 * i).unwrap();
            values.push(
                self.b
                    .ins()
                    .load(types::F32, MemFlagsData::trusted(), base, offset),
            );
        }

        for &instr in tape.instrs() {
            let result = self.translate_instr(instr, &values);
            values.push(result);
        }

        let result = *values.last().unwrap();
        self.b.ins().return_(&[result]);

        let config = self.module.target_config();
        self.b.finalize(config);
    }

    fn translate_instr(&mut self, instr: Instr, values: &[Value]) -> Value {
        let v = |id: ValueId| values[id.index()];

        match instr {
            Instr::I32Const(c) => self.b.ins().iconst(types::I32, c as i64),
            Instr::F32Const(c) => self.b.ins().f32const(c),

            Instr::I32Add(lhs, rhs) => self.b.ins().iadd(v(lhs), v(rhs)),
            Instr::I32Sub(lhs, rhs) => self.b.ins().isub(v(lhs), v(rhs)),
            Instr::I32Mul(lhs, rhs) => self.b.ins().imul(v(lhs), v(rhs)),

            Instr::F32FromI32(src) => self.b.ins().fcvt_from_sint(types::F32, v(src)),

            Instr::F32Neg(src) => self.b.ins().fneg(v(src)),
            Instr::F32Abs(src) => self.b.ins().fabs(v(src)),
            Instr::F32Add(lhs, rhs) => self.b.ins().fadd(v(lhs), v(rhs)),
            Instr::F32Sub(lhs, rhs) => self.b.ins().fsub(v(lhs), v(rhs)),
            Instr::F32Mul(lhs, rhs) => self.b.ins().fmul(v(lhs), v(rhs)),
            Instr::F32Div(lhs, rhs) => self.b.ins().fdiv(v(lhs), v(rhs)),

            Instr::F32Min(lhs, rhs) => self.b.ins().fmin(v(lhs), v(rhs)),
            Instr::F32Max(lhs, rhs) => self.b.ins().fmax(v(lhs), v(rhs)),

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
        let func_ref =
            *func_ref.get_or_insert_with(|| self.module.declare_func_in_func(*id, self.b.func));

        let inst = self.b.ins().call(func_ref, args);
        self.b.inst_results(inst)[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::Type;

    #[test]
    fn f32_ops() {
        let mut b = Tape::builder(vec![Type::F32, Type::F32]);
        let x = b.arg(0);
        let y = b.arg(1);
        let sum = b.instr(Instr::F32Add(x, y));
        let prod = b.instr(Instr::F32Mul(sum, x));
        let diff = b.instr(Instr::F32Sub(prod, y));
        let quot = b.instr(Instr::F32Div(diff, y));
        let neg = b.instr(Instr::F32Neg(quot));
        let abs = b.instr(Instr::F32Abs(neg));
        let min = b.instr(Instr::F32Min(abs, x));
        let max = b.instr(Instr::F32Max(min, y));
        let c = b.instr(Instr::F32Const(0.5));
        b.instr(Instr::F32Add(max, c));
        let tape = b.build().unwrap();

        let (x, y) = (1.75f32, -0.5f32);
        let expected = (-((x + y) * x - y) / y).abs().min(x).max(y) + 0.5;
        assert_eq!(Instance::new(&tape).evaluate(&[x, y]), expected);
    }

    #[test]
    fn f32_fn_calls() {
        let mut b = Tape::builder(vec![Type::F32, Type::F32]);
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
        let inst = Instance::new(&tape);

        let (x, y) = (0.7f32, 1.3f32);
        let expected = x.sin()
            + y.cos()
            + x.tan()
            + y.tan().recip()
            + x.exp()
            + y.ln()
            + x.log10()
            + x.powf(y);

        assert_eq!(inst.evaluate(&[x, y]), expected);
    }

    #[test]
    fn i32_ops() {
        let mut b = Tape::builder(vec![Type::F32]);
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
        let inst = Instance::new(&tape);

        let x = 1.5f32;
        assert_eq!(inst.evaluate(&[x]), 6.0 + x.powi(3));
    }

    #[test]
    fn clone_outlives_original() {
        let mut b = Tape::builder(vec![Type::F32]);
        let x = b.arg(0);
        let c = b.instr(Instr::F32Const(1.0));
        b.instr(Instr::F32Add(x, c));

        let tape = b.build().unwrap();
        let inst = Instance::new(&tape);
        let clone = inst.clone();
        drop(inst);

        assert_eq!(clone.evaluate(&[2.0]), 3.0);
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
