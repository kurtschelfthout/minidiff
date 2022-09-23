use std::ops::{Add, Mul};

// This module introduces the concept of forward mode AD,
// through overloading dual numbers.

// example function
fn f(t: f64) -> f64 {
    t.powi(2) + t + 1.0
}

// symbolically differentiated
fn df_sym(t: f64) -> f64 {
    2.0 * t + 1.0
}

// numerically differentiated
fn df_num(t: f64, h: f64) -> f64 {
    (f(t + h) - f(t)) / h
}

// "manual" automatic differentiation
fn f_ad(t: (f64, f64)) -> (f64, f64) {
    let (primalt, tangentt) = t;
    let (primal0, tangent0) = (primalt.powi(2), 2.0 * primalt * tangentt);
    let (primal1, tangent1) = (primal0 + primalt, tangent0 + 1.0);
    let (primal2, tangent2) = (primal1 + 1.0, tangent1);
    (primal2, tangent2)
}

// automatic differentiation through overloading
#[derive(Debug, Clone, Copy)]
struct Dual {
    primal: f64,
    tangent: f64,
}

impl Dual {
    fn constant(value: f64) -> Self {
        Dual {
            primal: value,
            tangent: 0.0,
        }
    }

    fn var(value: f64) -> Self {
        Dual {
            primal: value,
            tangent: 1.0,
        }
    }

    fn powi(self, exp: i32) -> Self {
        Dual {
            primal: self.primal.powi(exp),
            tangent: f64::from(exp) * self.primal.powi(exp - 1) * self.tangent,
        }
    }
}

impl Add for Dual {
    type Output = Dual;

    fn add(self, rhs: Self) -> Self::Output {
        Dual {
            primal: self.primal + rhs.primal,
            tangent: self.tangent + rhs.tangent,
        }
    }
}

impl Mul for Dual {
    type Output = Dual;

    fn mul(self, rhs: Self) -> Self::Output {
        Dual {
            primal: self.primal * rhs.primal,
            tangent: rhs.tangent * self.primal + self.tangent * rhs.primal,
        }
    }
}

fn f_ad_dual(t: Dual) -> Dual {
    t.powi(2) + t + Dual::constant(1.0)
}

pub fn run() {
    let t = 5.0;
    println!("Symbolic: {}", df_sym(t));

    let h = 0.000_001;
    println!("Numeric: {}", df_num(t, h));

    println!("Automatic: {}", f_ad((t, 1.0)).1);

    println!(
        "Automatic w/overloading: {}",
        f_ad_dual(Dual::var(t)).tangent
    );
}
