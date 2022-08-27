#![allow(clippy::toplevel_ref_arg)]

use std::{
    cell::RefCell,
    fmt::Display,
    ops::{Add, Mul},
    rc::Rc,
};

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

    fn sin(&self) -> Self {
        Dual::new(self.primal.sin(), self.tangent.scale(self.primal.cos()))
    }

    fn cos(&self) -> Self {
        Dual::new(self.primal.cos(), self.tangent.scale(-self.primal.sin()))
    }
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

// could do this instead - but using the vectorspace will work out better later
// impl D<f64> {
//     fn constant(value: f64) -> Self {
//         D {
//             reg: value,
//             deriv: 0.0,
//         }
//     }

//     fn sin(self) -> Self {
//         D {
//             reg: self.reg.sin(),
//             deriv: self.deriv * self.reg.cos(),
//         }
//     }
// }

// All four combinations of Forward + &Forward for ergonomics
impl<T: VectorSpace> Dual<T> {
    fn add_impl(&self, rhs: &Dual<T>) -> Dual<T> {
        Dual::new(self.primal + rhs.primal, self.tangent.add(&rhs.tangent))
    }
}

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

impl<T: VectorSpace> Dual<T> {
    fn mul_impl(&self, rhs: &Dual<T>) -> Dual<T> {
        Dual::new(
            self.primal * rhs.primal,
            rhs.tangent
                .scale(self.primal)
                .add(&self.tangent.scale(rhs.primal)),
        )
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

fn f<T: VectorSpace>(x: &Dual<T>) -> Dual<T> {
    let ref y = x.sin() * x;
    y.cos() + x * Dual::constant(10.0) + y * x.cos()
}

//wrapper
fn df(x: f64) -> f64 {
    let res = f(&Dual::new(x, 1.0));
    res.tangent
}

//mulitple outputs - works nicely
fn f_2out<T: VectorSpace>(x: &Dual<T>) -> (Dual<T>, Dual<T>) {
    let ref y = x.sin() * x.sin();
    (y + x * Dual::constant(10.0), x + y * Dual::constant(20.0))
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

// maybe also show full jacobian?

// multiple inputs - fix 1 - tupling
// note: copy here is necessary to allow the [T::zero(); N]
impl<T: Copy + VectorSpace, const N: usize> VectorSpace for [T; N] {
    fn zero() -> Self {
        [T::zero(); N]
    }

    fn add(&self, rhs: &Self) -> Self {
        let mut result = [T::zero(); N];
        for (i, (l, r)) in self.iter().zip(rhs).enumerate() {
            result[i] = l.add(r)
        }
        result
    }

    fn scale(&self, factor: f64) -> Self {
        let mut result = [T::zero(); N];
        for (i, v) in self.iter().enumerate() {
            result[i] = v.scale(factor)
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

// fix 2: introduce deltas
#[derive(Debug)]
enum Delta {
    Zero,
    OneHot(usize),
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

fn eval_delta<const N: usize>(x: f64, delta: &Delta, result: &mut [f64; N]) {
    match *delta {
        Delta::Zero => (),
        Delta::OneHot(i) => result[i] += x,
        Delta::Scale(factor, ref d2) => eval_delta(x * factor, d2, result),
        Delta::Add(ref l, ref r) => {
            eval_delta(x, l, result);
            eval_delta(x, r, result)
        }
    }
}

fn df_3in_v3(x: f64, y: f64, z: f64) -> [f64; 3] {
    let r = f_3in(
        &Dual::new(x, Rc::new(Delta::OneHot(0))),
        &Dual::new(y, Rc::new(Delta::OneHot(1))),
        &Dual::new(z, Rc::new(Delta::OneHot(2))),
    );
    let mut result = [0.0, 0.0, 0.0];
    eval_delta(1.0, &r.tangent, &mut result);
    result
}

// now note that we can do this evaluation both "forward" and "backward"
// and the "backward" evaluation above just transpires through the accumulators.
// is the equivalent "forward" eval maybe:
// Z -> [0,0,0]
// OH 1 -> [0,1,0]
// Sc(f,d) -> f * eval(d)
// Add(l,r) -> eval(l) + eval(r)
// which _still_ allocates a onehot vector for every input!
//  So where does the reverse in reverse-mode AD come from?
// it's all in the eval above?
// To work this out see what happens with a a 2 input - 2 output function
// do both forward and reverse AD and see where things happen?

// A last problem with this approach is that it doesn't take into account
// sharing. E.g. a program like:
fn f_sharing_bad<T: VectorSpace>(x: &Dual<T>, y: &Dual<T>, z: &Dual<T>) -> Dual<T> {
    let ref y = x.sin() * y.sin() * z * z; // some big rhs
    y + y
}

fn df_sharing_bad(x: f64, y: f64, z: f64) -> [f64; 3] {
    let r = f_sharing_bad(
        &Dual::new(x, Rc::new(Delta::OneHot(0))),
        &Dual::new(y, Rc::new(Delta::OneHot(1))),
        &Dual::new(z, Rc::new(Delta::OneHot(2))),
    );
    let mut result = [0.0, 0.0, 0.0];
    eval_delta(1.0, &r.tangent, &mut result);
    result
}

//if y = D v v' then y+y will result in a D (v+v) Add(v',v')
// v' is shared,but eval traverse v' twice!
// "a linear sized graph unravels to an exponentially larger tree" simonpj

// ok so we need the "monad" type
// Add/Mul become D<T> -> D<T> -> M<D<T>> where M is used to track the nodes
// f becomes D<T> -> ... > M<D<T>>

// since we don't have monads in rust, and anyway we'd prefer not to do the
// monadic source transformation, especially without do notation, let's try something else.
// all we really want is a "tape" - a record of the operations that have been executed
// so that we can aggregate the changes in reverse. The Let var = l in r setup simply creates
// that type as a linked list. But what if we change our D<T> to hold a normal vec, and just append
// every operation?

#[derive(Debug)]
enum DeltaVar {
    Zero,
    Var(usize),
    Scale(f64, Rc<DeltaVar>),
    Add(Rc<DeltaVar>, Rc<DeltaVar>),
}

#[derive(Debug)]
struct Tape {
    tape: RefCell<Vec<Rc<DeltaVar>>>,
}

impl Tape {
    fn new() -> Tape {
        Tape {
            tape: RefCell::new(vec![]),
        }
    }

    fn push_on_tape(&self, op: Dual<Rc<DeltaVar>>) -> Dual<Rc<DeltaVar>> {
        let mut tape_vec = self.tape.borrow_mut();
        let var = Dual {
            primal: op.primal,
            tangent: Rc::new(DeltaVar::Var(tape_vec.len())),
        };
        tape_vec.push(op.tangent);
        var
    }

    fn var(&self, primal: f64) -> DualTape {
        let var_x = Rc::new(DeltaVar::Var(self.tape.borrow().len()));
        self.tape.borrow_mut().push(Rc::clone(&var_x));
        DualTape {
            tape: self,
            dual: Dual {
                primal,
                tangent: Rc::clone(&var_x),
            },
        }
    }
}

#[derive(Debug)]
struct DualTape<'a> {
    tape: &'a Tape,
    dual: Dual<Rc<DeltaVar>>,
}

impl VectorSpace for Rc<DeltaVar> {
    fn zero() -> Self {
        Rc::new(DeltaVar::Zero)
    }

    fn add(&self, rhs: &Self) -> Self {
        Rc::new(DeltaVar::Add(Rc::clone(self), Rc::clone(rhs)))
    }

    fn scale(&self, factor: f64) -> Self {
        Rc::new(DeltaVar::Scale(factor, Rc::clone(self)))
    }
}

impl<'a> DualTape<'a> {
    // equivalent of deltaLet in Haskell
    fn delta_push(&self, op: Dual<Rc<DeltaVar>>) -> DualTape<'a> {
        let dual = self.tape.push_on_tape(op);
        DualTape {
            tape: self.tape,
            dual,
        }
    }

    fn add_impl(&self, rhs: &DualTape<'a>) -> DualTape<'a> {
        let op = &self.dual + &rhs.dual;
        self.delta_push(op)
    }

    fn mul_impl(&self, rhs: &DualTape<'a>) -> DualTape<'a> {
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

impl<'a> Add for DualTape<'a> {
    type Output = DualTape<'a>;

    fn add(self, rhs: Self) -> Self::Output {
        self.add_impl(&rhs)
    }
}

impl<'a> Add for &DualTape<'a> {
    type Output = DualTape<'a>;

    fn add(self, rhs: Self) -> Self::Output {
        self.add_impl(rhs)
    }
}

impl<'a> Add<&DualTape<'a>> for DualTape<'a> {
    type Output = DualTape<'a>;

    fn add(self, rhs: &DualTape<'a>) -> Self::Output {
        self.add_impl(rhs)
    }
}

impl<'a> Add<DualTape<'a>> for &DualTape<'a> {
    type Output = DualTape<'a>;

    fn add(self, rhs: DualTape<'a>) -> Self::Output {
        self.add_impl(&rhs)
    }
}

impl<'a> Mul for DualTape<'a> {
    type Output = DualTape<'a>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.mul_impl(&rhs)
    }
}

impl<'a> Mul for &DualTape<'a> {
    type Output = DualTape<'a>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.mul_impl(rhs)
    }
}

impl<'a> Mul<&DualTape<'a>> for DualTape<'a> {
    type Output = DualTape<'a>;

    fn mul(self, rhs: &DualTape<'a>) -> Self::Output {
        self.mul_impl(rhs)
    }
}

impl<'a> Mul<DualTape<'a>> for &DualTape<'a> {
    type Output = DualTape<'a>;

    fn mul(self, rhs: DualTape<'a>) -> Self::Output {
        self.mul_impl(&rhs)
    }
}

fn eval_deltavar(x: f64, deltavar: &DeltaVar, result: &mut Vec<f64>) {
    match *deltavar {
        DeltaVar::Zero => (),
        DeltaVar::Var(idx) => result[idx] += x,
        DeltaVar::Scale(factor, ref d2) => eval_deltavar(x * factor, d2, result),
        DeltaVar::Add(ref l, ref r) => {
            eval_deltavar(x, l, result);
            eval_deltavar(x, r, result);
        }
    }
}

fn eval(inputs: usize, dual_tape: DualTape) -> Vec<f64> {
    // this would be better as a map - we only use the input vars and a small
    // amount of intermediate vars (see also where in the call to eval_deltavar
    // where result is pop'ed.)
    let mut result = vec![0.0; dual_tape.tape.tape.borrow().len()];
    let mut tape = dual_tape.tape.tape.borrow_mut();
    // first seed the output type with 1.0 - this value is backpropagated
    eval_deltavar(1.0, &dual_tape.dual.tangent, &mut result);
    // now backpropagate by popping the tape - i.e. iterate in reverse
    while tape.len() > inputs {
        let deltavar = tape.pop().unwrap();
        let idx = tape.len();
        if result[idx] != 0.0 {
            eval_deltavar(result.pop().unwrap(), &deltavar, &mut result)
        }
    }
    result
}

fn f_sharing<'a>(x: &DualTape<'a>, y: &DualTape<'a>, z: &DualTape<'a>) -> DualTape<'a> {
    let ref r = x * y + z * x;
    r + r.cos() + r.sin()
}

fn df_sharing(x: f64, y: f64, z: f64) -> Vec<f64> {
    let tape = Tape::new();
    let x = &tape.var(x);
    let y = &tape.var(y);
    let z = &tape.var(z);
    let dual_tape = f_sharing(x, y, z);
    // println!("{dual_tape:?}");
    eval(3, dual_tape)
}

fn main() {
    let res = f(&Dual::new(10.0, 1.0));
    println!("{res:?}");

    let res = df(10.0);
    println!("{res:?}");

    let res = f_2out(&Dual::new(10.0, 1.0));
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

    let res = df_sharing_bad(10., 5.0, 1.0);
    println!("{res:?}");

    let res = df_sharing(1.0, 2.0, 3.0);
    print!("{res:?}")
}
