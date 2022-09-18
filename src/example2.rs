use std::rc::Rc;

use crate::ad::*;

fn f<T: VectorSpace>(x: &Dual<T>, y: &Dual<T>) -> Dual<T> {
    x * y
}

fn df_v1(x: f64, y: f64) -> (f64, f64) {
    (
        f(&Dual::new(x, 1.0), &Dual::constant(y)).tangent,
        f(&Dual::constant(x), &Dual::new(y, 1.0)).tangent,
    )
}

fn df_v2(x: f64, y: f64) -> [f64; 2] {
    f(&Dual::new(x, [1.0, 0.0]), &Dual::new(y, [0.0, 1.0])).tangent
}

fn df_v3(x: f64, y: f64) -> Vec<f64> {
    let dual_delta = f(
        &Dual::new(x, Rc::new(Delta::Var(0))),
        &Dual::new(y, Rc::new(Delta::Var(1))),
    );
    let mut result = vec![0.0, 0.0];
    eval_delta(1.0, &dual_delta.tangent, &mut result);
    result
}

fn f_sharing_bad<T: VectorSpace>(x: &Dual<T>, y: &Dual<T>) -> Dual<T> {
    let ref s = x * y;
    s + s
}

fn df_sharing_bad(x: f64, y: f64) -> Vec<f64> {
    let dual_delta = f_sharing_bad(
        &Dual::new(x, Rc::new(Delta::Var(0))),
        &Dual::new(y, Rc::new(Delta::Var(1))),
    );
    let mut result = vec![0.0, 0.0];
    eval_delta(1.0, &dual_delta.tangent, &mut result);
    result
}

fn f_sharing_fixed<'a>(x: &DualTrace<'a>, y: &DualTrace<'a>) -> DualTrace<'a> {
    let ref s = x * y;
    s + s
}

fn df_sharing_fixed(x: f64, y: f64) -> Vec<f64> {
    let trace = Trace::new();
    let x = &trace.var(x);
    let y = &trace.var(y);
    let dual_trace = f_sharing_fixed(x, y);
    eval(2, &dual_trace)
}

pub fn run() {
    let res = df_v1(3.0, 2.0);
    println!("{res:?}");

    let res = df_v2(3.0, 2.0);
    println!("{res:?}");

    let res = df_v3(3.0, 2.0);
    println!("{res:?}");

    let res = df_sharing_bad(3.0, 2.0);
    println!("{res:?}");

    let res = df_sharing_fixed(3.0, 2.0);
    println!("{res:?}");
}
