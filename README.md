# MiniDiff: A minimal reference implementation of automatic differentiation in Rust

Companion repo for [Automatic Differentiation: From Forward to Reverse in Small Steps](https://getcode.substack.com/p/automatic-differentiation-from-forward?r=1dboko&s=w&utm_campaign=post&utm_medium=web)

MiniDiff implements both forward and reverse mode automatic differentiation, and so enables differentiable programming in Rust.

It is meant to explain how automatic differentiation works, not as a crate to be used - though someone sufficiently motivated could develop it.

The development is largely based on <https://simon.peytonjones.org/provably-correct/>

## Short guide to the repo

Read the article linked above for full details, the tl;dr is:

- [playground.rs](https://github.com/kurtschelfthout/minidiff/blob/main/src/playground.rs): introduces the concept of AD. Standalone, also at this Rust playground: <https://play.rust-lang.org/?version=stable&mode=debug&edition=2021&gist=3575c2c0ce68498e6f1528855c364e7c>
- [ad.rs](https://github.com/kurtschelfthout/minidiff/blob/main/src/ad.rs): the implementations, and the intermediate steps from forward to reverse mode AD. If you're only interested in the "finished products" (i.e. not the intermediate steps, certainly don't mean they're production ready whatsoever):
  - [The code marked as Step 1](https://github.com/kurtschelfthout/minidiff/blob/main/src/ad.rs#L8-L151) is forward mode AD
  - [The code marked as Step 3 and Step 4](https://github.com/kurtschelfthout/minidiff/blob/main/src/ad.rs#L178-L377) *together* is reverse mode AD.
- [example1.rs](https://github.com/kurtschelfthout/minidiff/blob/main/src/example1.rs) and [example2.rs](https://github.com/kurtschelfthout/minidiff/blob/main/src/example2.rs): Examples of usage of the various steps. Again usage of the "finished products":
  - [Usage of forward mode AD](https://github.com/kurtschelfthout/minidiff/blob/main/src/example1.rs#L5-L37)

    ```rust
    fn f<T: VectorSpace>(t: &Dual<T>) -> Dual<T> {
        t.powi(2) + t + Dual::constant(1.0)
    }

    let res = f(&Dual::new(10.0, 1.0));
    println!("{res:?}");

    // prints Dual { primal: 111.0, tangent: 21.0 }
    ```

  - [Usage of reverse mode AD](https://github.com/kurtschelfthout/minidiff/blob/main/src/example1.rs#L97-L104) and [here in example2.rs](https://github.com/kurtschelfthout/minidiff/blob/main/src/example2.rs#L45-L56)

    ```rust
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

    let res = df_sharing_fixed(3.0, 2.0);
    println!("{res:?}");

    // prints [4.0, 6.0]
    ```