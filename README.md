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
  - [Usage of reverse mode AD](https://github.com/kurtschelfthout/minidiff/blob/main/src/example1.rs#L97-L104) and [here in example2.rs](https://github.com/kurtschelfthout/minidiff/blob/main/src/example2.rs#L45-L56)
