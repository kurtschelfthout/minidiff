#[derive(Debug, Clone)]
struct D<T> {
    reg: f64,
    deriv: T,
}

impl<T> Add for D<T>
where
    T: Clone + Add<T, Output = T>,
{
    type Output = D<T>;

    fn add(self, D { reg, deriv }: Self) -> Self::Output {
        D {
            reg: self.reg + reg,
            deriv: self.deriv + deriv,
        }
    }
}

impl<T> Add for &D<T>
where
    T: Clone + Add<T, Output = T>,
{
    type Output = D<T>;

    fn add(self, D { reg, deriv }: Self) -> Self::Output {
        D {
            reg: self.reg + reg,
            deriv: self.deriv.clone() + deriv.clone(),
        }
    }
}

impl<T> Add<D<T>> for &D<T>
where
    T: Clone + Add<T, Output = T>,
{
    type Output = D<T>;

    fn add(self, D { reg, deriv }: D<T>) -> Self::Output {
        D {
            reg: self.reg + reg,
            deriv: self.deriv.clone() + deriv,
        }
    }
}

impl<T> Add<&D<T>> for D<T>
where
    T: Clone + Add<T, Output = T>,
{
    type Output = D<T>;

    fn add(self, D { reg, deriv }: &D<T>) -> Self::Output {
        D {
            reg: self.reg + reg,
            deriv: self.deriv + deriv.clone(),
        }
    }
}

fn f<T>(x: &D<T>) -> D<T>
where
    T: Clone + Add<T, Output = T>,
{
    let ref y = x + x;
    y + x + y + x
}

fn main() {
    let res = f(&D {
        reg: 10.0,
        deriv: 1.0,
    });
    println!("{res:?}");
}

#[derive(Debug, Clone)]
enum DeltaLet {
    Zero,
    Var(usize),
    Scale(f64, Rc<DeltaLet>),
    Add(Rc<DeltaLet>, Rc<DeltaLet>),
    Let(usize, Rc<DeltaLet>, Rc<DeltaLet>),
}

impl VectorSpace for Rc<DeltaLet> {
    fn zero() -> Self {
        Rc::new(DeltaLet::Zero)
    }

    fn add(self, rhs: Self) -> Self {
        Rc::new(DeltaLet::Add(self, rhs))
    }

    fn scale(self, factor: f64) -> Self {
        Rc::new(DeltaLet::Scale(factor, self))
    }
}

fn f_sharing_painful(
    x: &D<Rc<DeltaLet>>,
    y: &D<Rc<DeltaLet>>,
    z: &D<Rc<DeltaLet>>,
) -> (D<Rc<DeltaLet>>, Vec<&DeltaLet>) {
    // let ref r = x * z + y * z;
    // r + r

    let var_ids = AtomicUsize::new(0);
    let mut delta_map: Vec<&DeltaLet> = vec![];

    let var_id = var_ids.fetch_add(1, Ordering::SeqCst);
    let ref x_t_z = x * z;
    delta_map.insert(var_id, &x_t_z.deriv);

    let var_id = var_ids.fetch_add(1, Ordering::SeqCst);
    let ref y_t_z = y * z;
    delta_map.insert(var_id, &y_t_z.deriv);

    let r_var = var_ids.fetch_add(1, Ordering::SeqCst);
    let ref r = x_t_z * y_t_z;
    delta_map.insert(r_var, &r.deriv);

    let r_var = var_ids.fetch_add(1, Ordering::SeqCst);
    let r_p_r = r + r;
    delta_map.insert(r_var, &r_p_r.deriv);

    (r_p_r, delta_map);

    let mut hasher = DefaultHasher::new();
    let a = Rc::new(Delta::Zero);
    std::ptr::hash(a.as_ref(), &mut hasher);
    let actual = hasher.finish();
    println!("{actual:?}");

    let mut hasher = DefaultHasher::new();
    let b = Rc::new(Delta::Zero);
    std::ptr::hash(b.as_ref(), &mut hasher);
    let actual = hasher.finish();
    println!("{actual:?}");
}

// #[derive(Debug)]
// struct M {
//     delta_id: RefCell<usize>,
//     delta_map: RefCell<Vec<Rc<Delta>>>,
// }

// #[derive(Debug)]
// struct Wrapper {
//     w: Rc<Delta>,
// }

// impl Hash for Wrapper {
//     fn hash<H: Hasher>(&self, state: &mut H) {
//         std::ptr::hash(self.w.as_ref(), state);
//     }
// }

// impl PartialEq for Wrapper {
//     fn eq(&self, other: &Self) -> bool {
//         std::ptr::eq(self.w.as_ref(), other.w.as_ref())
//     }
// }

// impl Eq for Wrapper {}

// #[derive(Debug)]
// struct DeltaBinding {
//     delta_map: RefCell<HashMap<Wrapper, f64>>,
//     delta: Rc<Delta>,
// }

// impl VectorSpace for DeltaBinding {
//     fn zero() -> Self {
//         DeltaBinding {
//             delta_map: RefCell::new(HashMap::new()),
//             delta: Rc::new(Delta::Zero),
//         }
//     }

//     fn add(self, rhs: Self) -> Self {
//         todo!()
//     }

//     fn scale(self, factor: f64) -> Self {
//         let mut hm = self.delta_map.get_mut();
//         let delta = Rc::new(Delta::Scale(factor, self.delta));
//         hm.insert(Wrapper { w: delta }, 0.0);
//         &mut DeltaBinding {
//             delta_map: self.delta_map,
//             delta: delta,
//         }
//     }
// }
