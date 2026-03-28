use crate::transforms::Transform;

/// Wraps a [`Transform`] that applies to a scalar so that it applies
/// to the first ("x") coordinate in a pair instead.
pub struct XTransform<R>(pub R);
impl<A: 'static, B: 'static, R: Transform<A>> Transform<(A, B)> for XTransform<R> {
    fn apply<I: ?Sized>(&self, data: &mut I)
    where
        for<'a> &'a mut I: IntoIterator<Item = &'a mut (A, B)>,
    {
        // We need a full function, not just a closure, to force the lifetime to match.
        fn first_mut<A, B>(pair: &mut (A, B)) -> &mut A {
            &mut pair.0
        }

        let mut mapped = IntoIterMutMap(data, first_mut::<A, B>);
        self.0.apply::<IntoIterMutMap<'_, I, _>>(&mut mapped);
    }
}

/// Wraps a [`Transform`] that applies to a scalar so that it applies
/// to the second ("y") coordinate in a pair instead.
///
/// Note that you usually don't need to do this yourself, as [`Transformable`]
/// on pairs defaults to applying this way.
pub struct YTransform<R>(pub R);
impl<A: 'static, B: 'static, R: Transform<B>> Transform<(A, B)> for YTransform<R> {
    fn apply<I: ?Sized>(&self, data: &mut I)
    where
        for<'a> &'a mut I: IntoIterator<Item = &'a mut (A, B)>,
    {
        // We need a full function, not just a closure, to force the lifetime to match.
        fn second_mut<A, B>(pair: &mut (A, B)) -> &mut B {
            &mut pair.1
        }

        let mut mapped = IntoIterMutMap(data, second_mut::<A, B>);
        self.0.apply::<IntoIterMutMap<'_, I, _>>(&mut mapped);
    }
}

struct IntoIterMutMap<'inner, I: ?Sized, F>(&'inner mut I, F);
impl<'outer, I: ?Sized, F, A: 'outer, B: 'outer> IntoIterator
    for &'outer mut IntoIterMutMap<'_, I, F>
where
    &'outer mut I: IntoIterator<Item = &'outer mut A>,
    F: Fn(&mut A) -> &mut B,
{
    type IntoIter = std::iter::Map<<&'outer mut I as IntoIterator>::IntoIter, &'outer mut F>;
    type Item = &'outer mut B;
    fn into_iter(self) -> Self::IntoIter {
        IntoIterator::into_iter(&mut *self.0).map(&mut self.1)
    }
}
