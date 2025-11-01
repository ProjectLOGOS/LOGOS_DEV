/**
 * Trinitarian Mandelbrot Equation derived from theological mathematics
 *
 * Scaffold + operational code
 */
class Complex {
  constructor(real, imag) {
    this.real = real;
    this.imag = imag;
  }
  add(o) {
    if (typeof o === "number") return new Complex(this.real + o, this.imag);
    return new Complex(
      this.real * o.real - this.imag * o.imag + o.real,
      this.real * o.imag + this.imag * o.real
    );
  }
  pow(n) {
    let r = new Complex(1, 0);
    for (let i = 0; i < n; i++) r = r.multiply(this);
    return r;
  }
  multiply(o) {
    const real = this.real * o.real - this.imag * o.imag;
    const imag = this.real * o.imag + this.imag * o.real;
    return new Complex(real, imag);
  }
  abs() {
    return Math.hypot(this.real, this.imag);
  }
}

function trinitarianMandelbrot(cIn, maxIter=100, escapeR=2) {
  const c = typeof cIn==="number"? new Complex(cIn,0): cIn;
  let z = new Complex(0,0);
  for (let i=0; i<maxIter; i++){
    z = z.pow(3).add(z.pow(2)).add(z).add(c);
    if (z.abs()>escapeR) return {escaped:true, iter:i};
  }
  return {escaped:false, iter:maxIter};
}

console.log(trinitarianMandelbrot(new Complex(0,0)));
