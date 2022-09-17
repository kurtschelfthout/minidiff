#![allow(clippy::toplevel_ref_arg)]
mod playground;

use std::{
    cell::{RefCell, RefMut},
    fmt::Display,
    ops::{Add, Mul},
    rc::Rc,
};

// ------- Step 1: Dual<T> for T=f64 -----------

#[derive(Debug)]
struct Dual<T> {
    primal: f64,
    tangent: T,
}

impl<T> Display for Dual<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.primal, self.tangent,)
    }
}

trait VectorSpace {
    fn zero() -> Self;
    fn add(&self, rhs: &Self) -> Self;
    fn scale(&self, factor: f64) -> Self;
}

impl VectorSpace for f64 {
    fn zero() -> Self {
        0.0
    }

    fn add(&self, rhs: &Self) -> Self {
        self + rhs
    }

    fn scale(&self, factor: f64) -> Self {
        self * factor
    }
}

impl<T: VectorSpace> Dual<T> {
    fn constant(primal: f64) -> Self {
        Dual {
            primal,
            tangent: T::zero(),
        }
    }

    fn new(primal: f64, tangent: T) -> Self {
        Dual { primal, tangent }
    }

    fn chain(&self, primal: f64, factor: f64) -> Self {
        Dual::new(primal, self.tangent.scale(factor))
    }

    fn sin(&self) -> Self {
        self.chain(self.primal.sin(), self.primal.cos())
    }

    fn cos(&self) -> Self {
        self.chain(self.primal.cos(), -self.primal.sin())
    }

    fn powi(&self, n: i32) -> Self {
        self.chain(self.primal.powi(n), f64::from(n) * self.primal.powi(n - 1))
    }

    fn add_impl(&self, rhs: &Dual<T>) -> Dual<T> {
        Dual::new(self.primal + rhs.primal, self.tangent.add(&rhs.tangent))
    }

    fn mul_impl(&self, rhs: &Dual<T>) -> Dual<T> {
        Dual::new(
            self.primal * rhs.primal,
            rhs.tangent
                .scale(self.primal)
                .add(&self.tangent.scale(rhs.primal)),
        )
    }
}

// All four combinations of Dual + &Dual for ergonomics

impl<T: VectorSpace> Add for Dual<T> {
    type Output = Dual<T>;

    fn add(self, rhs: Self) -> Self::Output {
        self.add_impl(&rhs)
    }
}

impl<T: VectorSpace> Add for &Dual<T> {
    type Output = Dual<T>;

    fn add(self, rhs: Self) -> Self::Output {
        self.add_impl(rhs)
    }
}

impl<T: VectorSpace> Add<&Dual<T>> for Dual<T> {
    type Output = Dual<T>;

    fn add(self, rhs: &Dual<T>) -> Self::Output {
        self.add_impl(rhs)
    }
}

impl<T: VectorSpace> Add<Dual<T>> for &Dual<T> {
    type Output = Dual<T>;

    fn add(self, rhs: Dual<T>) -> Self::Output {
        self.add_impl(&rhs)
    }
}

impl<T: VectorSpace> Mul for Dual<T> {
    type Output = Dual<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.mul_impl(&rhs)
    }
}

impl<T: VectorSpace> Mul for &Dual<T> {
    type Output = Dual<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.mul_impl(rhs)
    }
}

impl<T: VectorSpace> Mul<&Dual<T>> for Dual<T> {
    type Output = Dual<T>;

    fn mul(self, rhs: &Dual<T>) -> Self::Output {
        self.mul_impl(rhs)
    }
}

impl<T: VectorSpace> Mul<Dual<T>> for &Dual<T> {
    type Output = Dual<T>;

    fn mul(self, rhs: Dual<T>) -> Self::Output {
        self.mul_impl(&rhs)
    }
}

fn f<T: VectorSpace>(t: &Dual<T>) -> Dual<T> {
    t.powi(2) + t + Dual::constant(1.0)
}

//wrapper
fn df(t: f64) -> f64 {
    let res = f(&Dual::new(t, 1.0));
    res.tangent
}

//mulitple outputs - works nicely
fn f_2out<T: VectorSpace>(t: &Dual<T>) -> (Dual<T>, Dual<T>) {
    let x = t.powi(2) + t + Dual::constant(1.0);
    let y = t.powi(2) + Dual::constant(3.0) * t + Dual::constant(2.0);
    (x, y)
}

fn df_2out(x: f64) -> (f64, f64) {
    let (dx, dy) = f_2out(&Dual::new(x, 1.0));
    (dx.tangent, dy.tangent)
}

//multiple inputs: bad - evaluates function multiple times!
fn f_3in<T: VectorSpace>(x: &Dual<T>, y: &Dual<T>, z: &Dual<T>) -> Dual<T> {
    x * y.sin() + z * y
}

fn df_3in_v1(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let r1 = f_3in(&Dual::new(x, 1.0), &Dual::constant(y), &Dual::constant(z));
    let r2 = f_3in(&Dual::constant(x), &Dual::new(y, 1.0), &Dual::constant(z));
    let r3 = f_3in(&Dual::constant(x), &Dual::constant(y), &Dual::new(z, 1.0));
    (r1.tangent, r2.tangent, r3.tangent)
}

// "full jacobian"
// fn f_3in3out<T: VectorSpace>(x: &Dual<T>, y: &Dual<T>, z: &Dual<T>) -> (Dual<T>, Dual<T>, Dual<T>) {
//     (x * y, x * z, y * z)
// }

// fn df_3in3out(x: f64, y: f64, z: f64) -> ((f64, f64, f64), (f64, f64, f64), (f64, f64, f64)) {
//     let (r11, r12, r13) = f_3in3out(&Dual::new(x, 1.0), &Dual::constant(y), &Dual::constant(z));
//     let (r21, r22, r23) = f_3in3out(&Dual::constant(x), &Dual::new(y, 1.0), &Dual::constant(z));
//     let (r31, r32, r33) = f_3in3out(&Dual::constant(x), &Dual::constant(y), &Dual::new(z, 1.0));
//     (
//         (r11.tangent, r12.tangent, r13.tangent),
//         (r21.tangent, r22.tangent, r23.tangent),
//         (r31.tangent, r32.tangent, r33.tangent),
//     )
// }

// ------- Step 2: Dual<T> for T=Vec<f64> -----------

// note: copy here is necessary to allow the [T::zero(); N]
impl<T: Copy + VectorSpace, const N: usize> VectorSpace for [T; N] {
    fn zero() -> Self {
        [T::zero(); N]
    }

    fn add(&self, rhs: &Self) -> Self {
        let mut result = [T::zero(); N];
        for (i, (l, r)) in self.iter().zip(rhs).enumerate() {
            result[i] = l.add(r);
        }
        result
    }

    fn scale(&self, factor: f64) -> Self {
        let mut result = [T::zero(); N];
        for (i, v) in self.iter().enumerate() {
            result[i] = v.scale(factor);
        }
        result
    }
}

// much better - f only called once - but memory is quadratic n_inputs x n_inputs
fn df_3in_v2(x: f64, y: f64, z: f64) -> [f64; 3] {
    let r = f_3in(
        &Dual::new(x, [1.0, 0.0, 0.0]),
        &Dual::new(y, [0.0, 1.0, 0.0]),
        &Dual::new(z, [0.0, 0.0, 1.0]),
    );
    r.tangent
}

// ------- Step 3: Dual<T> for T=Delta -----------

#[derive(Debug)]
enum Delta {
    Zero,
    Var(usize),
    Scale(f64, Rc<Delta>),
    Add(Rc<Delta>, Rc<Delta>),
}

impl VectorSpace for Rc<Delta> {
    fn zero() -> Self {
        Rc::new(Delta::Zero)
    }

    fn add(&self, rhs: &Self) -> Self {
        Rc::new(Delta::Add(Rc::clone(self), Rc::clone(rhs)))
    }

    fn scale(&self, factor: f64) -> Self {
        Rc::new(Delta::Scale(factor, Rc::clone(self)))
    }
}

fn eval_delta(scale_acc: f64, delta: &Delta, result: &mut Vec<f64>) {
    match *delta {
        Delta::Zero => (),
        Delta::Var(i) => result[i] += scale_acc,
        Delta::Scale(factor, ref d2) => eval_delta(scale_acc * factor, d2, result),
        Delta::Add(ref l, ref r) => {
            eval_delta(scale_acc, l, result);
            eval_delta(scale_acc, r, result);
        }
    }
}

fn df_3in_v3(x: f64, y: f64, z: f64) -> Vec<f64> {
    let r = f_3in(
        &Dual::new(x, Rc::new(Delta::Var(0))),
        &Dual::new(y, Rc::new(Delta::Var(1))),
        &Dual::new(z, Rc::new(Delta::Var(2))),
    );
    let mut result = vec![0.0, 0.0, 0.0];
    eval_delta(1.0, &r.tangent, &mut result);
    result
}

// ------- Step 4: DualTrace<T> for T=Delta -----------

fn f_sharing_bad<T: VectorSpace>(x: &Dual<T>, y: &Dual<T>, z: &Dual<T>) -> Dual<T> {
    let ref y = x.sin() * y.sin() * z * z; // some big rhs
    y + y
}

fn df_sharing_bad(x: f64, y: f64, z: f64) -> Vec<f64> {
    let r = f_sharing_bad(
        &Dual::new(x, Rc::new(Delta::Var(0))),
        &Dual::new(y, Rc::new(Delta::Var(1))),
        &Dual::new(z, Rc::new(Delta::Var(2))),
    );
    let mut result = vec![0.0, 0.0, 0.0];
    eval_delta(1.0, &r.tangent, &mut result);
    result
}

#[derive(Debug)]
struct Trace {
    trace: RefCell<Vec<Rc<Delta>>>,
}

impl Trace {
    fn new() -> Trace {
        Trace {
            trace: RefCell::new(vec![]),
        }
    }

    fn push(&self, op: Dual<Rc<Delta>>) -> Dual<Rc<Delta>> {
        let mut trace = self.trace.borrow_mut();
        let var = Dual {
            primal: op.primal,
            tangent: Rc::new(Delta::Var(trace.len())),
        };
        trace.push(op.tangent);
        var
    }

    fn var(&self, primal: f64) -> DualTrace {
        let var_x = Rc::new(Delta::Var(self.trace.borrow().len()));
        self.trace.borrow_mut().push(Rc::clone(&var_x));
        DualTrace {
            trace: self,
            dual: Dual {
                primal,
                tangent: Rc::clone(&var_x),
            },
        }
    }
}

#[derive(Debug)]
struct DualTrace<'a> {
    trace: &'a Trace,
    dual: Dual<Rc<Delta>>,
}

impl<'a> DualTrace<'a> {
    fn trace_len(&self) -> usize {
        self.trace.trace.borrow().len()
    }

    fn trace_mut(&self) -> RefMut<Vec<Rc<Delta>>> {
        self.trace.trace.borrow_mut()
    }

    // equivalent of deltaLet in Haskell
    fn delta_push(&self, op: Dual<Rc<Delta>>) -> DualTrace<'a> {
        let dual = self.trace.push(op);
        DualTrace {
            trace: self.trace,
            dual,
        }
    }

    fn add_impl(&self, rhs: &DualTrace<'a>) -> DualTrace<'a> {
        let op = &self.dual + &rhs.dual;
        self.delta_push(op)
    }

    fn mul_impl(&self, rhs: &DualTrace<'a>) -> DualTrace<'a> {
        let op = &self.dual * &rhs.dual;
        self.delta_push(op)
    }

    fn sin(&self) -> Self {
        let op = self.dual.sin();
        self.delta_push(op)
    }

    fn cos(&self) -> Self {
        let op = self.dual.cos();
        self.delta_push(op)
    }
}

impl<'a> Add for DualTrace<'a> {
    type Output = DualTrace<'a>;

    fn add(self, rhs: Self) -> Self::Output {
        self.add_impl(&rhs)
    }
}

impl<'a> Add for &DualTrace<'a> {
    type Output = DualTrace<'a>;

    fn add(self, rhs: Self) -> Self::Output {
        self.add_impl(rhs)
    }
}

impl<'a> Add<&DualTrace<'a>> for DualTrace<'a> {
    type Output = DualTrace<'a>;

    fn add(self, rhs: &DualTrace<'a>) -> Self::Output {
        self.add_impl(rhs)
    }
}

impl<'a> Add<DualTrace<'a>> for &DualTrace<'a> {
    type Output = DualTrace<'a>;

    fn add(self, rhs: DualTrace<'a>) -> Self::Output {
        self.add_impl(&rhs)
    }
}

impl<'a> Mul for DualTrace<'a> {
    type Output = DualTrace<'a>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.mul_impl(&rhs)
    }
}

impl<'a> Mul for &DualTrace<'a> {
    type Output = DualTrace<'a>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.mul_impl(rhs)
    }
}

impl<'a> Mul<&DualTrace<'a>> for DualTrace<'a> {
    type Output = DualTrace<'a>;

    fn mul(self, rhs: &DualTrace<'a>) -> Self::Output {
        self.mul_impl(rhs)
    }
}

impl<'a> Mul<DualTrace<'a>> for &DualTrace<'a> {
    type Output = DualTrace<'a>;

    fn mul(self, rhs: DualTrace<'a>) -> Self::Output {
        self.mul_impl(&rhs)
    }
}

fn eval(inputs: usize, dual_trace: &DualTrace) -> Vec<f64> {
    // this would be better as a map - we only use the input vars and a small
    // amount of intermediate vars (see also where in the call to eval_delta_vec
    // where result is pop'ed.)
    let mut result = vec![0.0; dual_trace.trace_len()];
    let mut trace = dual_trace.trace_mut();
    // first seed the output type with 1.0 - this value is backpropagated
    eval_delta(1.0, &dual_trace.dual.tangent, &mut result);
    // now backpropagate by popping the trace - i.e. iterate in reverse
    while trace.len() > inputs {
        let deltavar = trace.pop().unwrap();
        let idx = trace.len();
        if result[idx] != 0.0 {
            eval_delta(result.pop().unwrap(), &deltavar, &mut result);
        }
    }
    result
}

fn f_sharing<'a>(x: &DualTrace<'a>, y: &DualTrace<'a>, z: &DualTrace<'a>) -> DualTrace<'a> {
    let ref y = x.sin() * y.sin() * z * z; // some big rhs
    y + y
}

fn df_sharing(x: f64, y: f64, z: f64) -> Vec<f64> {
    let trace = Trace::new();
    let x = &trace.var(x);
    let y = &trace.var(y);
    let z = &trace.var(z);
    let dual_trace = f_sharing(x, y, z);
    eval(3, &dual_trace)
}

fn g<T: VectorSpace>(x: &Dual<T>, y: &Dual<T>) -> Dual<T> {
    x * y
}

fn dg_v1(x: f64, y: f64) -> (f64, f64) {
    (
        g(&Dual::new(x, 1.0), &Dual::constant(y)).tangent,
        g(&Dual::constant(x), &Dual::new(y, 1.0)).tangent,
    )
}

fn dg_v2(x: f64, y: f64) -> [f64; 2] {
    g(&Dual::new(x, [1.0, 0.0]), &Dual::new(y, [0.0, 1.0])).tangent
}

fn dg_v3(x: f64, y: f64) -> Vec<f64> {
    let dual_delta = g(
        &Dual::new(x, Rc::new(Delta::Var(0))),
        &Dual::new(y, Rc::new(Delta::Var(1))),
    );
    let mut result = vec![0.0, 0.0];
    eval_delta(1.0, &dual_delta.tangent, &mut result);
    result
}

fn g_sharing_bad<T: VectorSpace>(x: &Dual<T>, y: &Dual<T>) -> Dual<T> {
    let ref s = x * y;
    s + s
}

fn dg_sharing_bad(x: f64, y: f64) -> Vec<f64> {
    let dual_delta = g_sharing_bad(
        &Dual::new(x, Rc::new(Delta::Var(0))),
        &Dual::new(y, Rc::new(Delta::Var(1))),
    );
    let mut result = vec![0.0, 0.0];
    eval_delta(1.0, &dual_delta.tangent, &mut result);
    result
}

fn g_sharing_fixed<'a>(x: &DualTrace<'a>, y: &DualTrace<'a>) -> DualTrace<'a> {
    let ref s = x * y;
    s + s
}

fn dg_sharing_fixed(x: f64, y: f64) -> Vec<f64> {
    let trace = Trace::new();
    let x = &trace.var(x);
    let y = &trace.var(y);
    let dual_trace = g_sharing_fixed(x, y);
    eval(2, &dual_trace)
}

fn main() {
    playground::main();

    let res = f(&Dual::new(10.0, 1.0));
    println!("{res:?}");

    let res = df(10.0);
    println!("{res:?}");

    let res = f_2out(&Dual::new(10.0, 1.0));
    println!("{res:?}");

    let res = df_2out(10.0);
    println!("{res:?}");

    let res = f_3in(
        &Dual::new(10.0, 1.0),
        &Dual::constant(5.0),
        &Dual::constant(1.0),
    );
    println!("{res:?}");

    let res = df_3in_v1(10.0, 5.0, 1.0);
    println!("{res:?}");

    let res = df_3in_v2(10.0, 5.0, 1.0);
    println!("{res:?}");

    let res = df_3in_v3(10.0, 5.0, 1.0);
    println!("{res:?}");

    let res = df_sharing_bad(1.0, 2.0, 3.0);
    println!("{res:?}");

    let res = df_sharing(1.0, 2.0, 3.0);
    println!("{res:?}");

    let res = dg_v1(3.0, 2.0);
    println!("{res:?}");

    let res = dg_v2(3.0, 2.0);
    println!("{res:?}");

    let res = dg_v3(3.0, 2.0);
    println!("{res:?}");

    let res = dg_sharing_bad(3.0, 2.0);
    println!("{res:?}");

    let res = dg_sharing_fixed(3.0, 2.0);
    println!("{res:?}");
}
