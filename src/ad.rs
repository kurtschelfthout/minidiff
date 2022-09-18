use std::{
    cell::{RefCell, RefMut},
    fmt::Display,
    ops::{Add, Mul},
    rc::Rc,
};

// ------- Step 1: Dual<T> for T=f64 -----------

#[derive(Debug)]
pub struct Dual<T> {
    pub primal: f64,
    pub tangent: T,
}

impl<T> Display for Dual<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.primal, self.tangent,)
    }
}

pub trait VectorSpace {
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
    pub fn constant(primal: f64) -> Self {
        Dual {
            primal,
            tangent: T::zero(),
        }
    }

    pub fn new(primal: f64, tangent: T) -> Self {
        Dual { primal, tangent }
    }

    fn chain(&self, primal: f64, factor: f64) -> Self {
        Dual::new(primal, self.tangent.scale(factor))
    }

    pub fn sin(&self) -> Self {
        self.chain(self.primal.sin(), self.primal.cos())
    }

    pub fn cos(&self) -> Self {
        self.chain(self.primal.cos(), -self.primal.sin())
    }

    pub fn powi(&self, n: i32) -> Self {
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

// ------- Step 3: Dual<T> for T=Delta -----------

#[derive(Debug)]
pub enum Delta {
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

pub fn eval_delta(scale_acc: f64, delta: &Delta, result: &mut Vec<f64>) {
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

// ------- Step 4: DualTrace<T> for T=Delta -----------

#[derive(Debug)]
pub struct Trace {
    trace: RefCell<Vec<Rc<Delta>>>,
}

impl Trace {
    pub fn new() -> Trace {
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

    pub fn var(&self, primal: f64) -> DualTrace {
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
pub struct DualTrace<'a> {
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

    pub fn sin(&self) -> Self {
        let op = self.dual.sin();
        self.delta_push(op)
    }

    pub fn cos(&self) -> Self {
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

pub fn eval(inputs: usize, dual_trace: &DualTrace) -> Vec<f64> {
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
