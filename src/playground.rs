use std::ops::{Add, Mul};

fn f(t: f64) -> f64 {
    t.powi(2) + t + 1.0
}

fn df(t: f64) -> f64 {
    2.0 * t + 1.0
}

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

fn f_ad(t: Dual) -> Dual {
    t.powi(2) + t + Dual::constant(1.0)
}

pub fn main() {
    let t = 5.0;
    println!("Symbolic: {}", df(t));

    let h = 0.000_001;
    println!("Numeric: {}", (f(t + h) - f(t)) / h);

    println!("Automatic: {}", f_ad(Dual::var(t)).tangent);
}
