// Package half provides half-precision floating-point types.
package half

// Float defines the common interface for half-precision floating-point types.
// Method names align with math/big.Float and math package conventions.
type Float interface {
	// Float32 returns the float32 representation.
	Float32() float32

	// Float64 returns the float64 representation.
	Float64() float64

	// Bits returns the IEEE 754 binary representation as uint16.
	Bits() uint16

	// IsNaN reports whether f is NaN.
	IsNaN() bool

	// IsInf reports whether f is infinity.
	// sign > 0: +Inf, sign < 0: -Inf, sign == 0: either.
	IsInf(sign int) bool

	// IsFinite reports whether f is finite (not Inf or NaN).
	IsFinite() bool

	// IsNormal reports whether f is normal (not zero, subnormal, Inf, or NaN).
	IsNormal() bool

	// Sign returns -1 if f < 0, 0 if f == 0 or NaN, +1 if f > 0.
	Sign() int

	// Signbit reports whether f is negative or negative zero.
	Signbit() bool

	// String returns the decimal representation.
	String() string
}
