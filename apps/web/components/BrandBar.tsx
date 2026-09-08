/**
 * Co-branding bar: Lux Sanans (left) and Quantanite (right), on screen for the
 * whole session including the call.
 *
 * The Quantanite asset is a pure-black wordmark (verified: a single #000000
 * colour on transparency), which is invisible on this dark UI — `.is-dark-mark`
 * inverts it to crisp white. Swap in a light-variant PNG and drop that class if
 * one becomes available.
 */
export function BrandBar() {
  return (
    <header className="brandbar">
      <img
        className="brandbar-logo brandbar-lux"
        src="/logos/lux-sanans.png"
        alt="Lux Sanans"
        width={760}
        height={117}
      />
      <span className="brandbar-x" aria-hidden="true" />
      <img
        className="brandbar-logo brandbar-quantanite is-dark-mark"
        src="/logos/quantanite.png"
        alt="Quantanite"
        width={512}
        height={107}
      />
    </header>
  );
}
