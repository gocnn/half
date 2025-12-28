package bfloat16

import (
	"math"
	"strconv"

	"github.com/gocnn/half"
)

// Compile-time interface compliance check.
var _ half.Float = BFloat16(0)

// BFloat16 represents Brain Floating Point 16-bit numbers.
// Layout: 1 sign + 8 exponent + 7 mantissa (same exponent as float32).
type BFloat16 uint16

// SmallestNonzero is the smallest positive nonzero BFloat16.
const SmallestNonzero = BFloat16(0x0001)

// BFloat16 bit layout.
const (
	signMask BFloat16 = 0x8000
	expMask  BFloat16 = 0x7f80
	coefMask BFloat16 = 0x007f
	qnanBit  BFloat16 = 0x0040
	uvPosInf BFloat16 = 0x7f80
	uvNegInf BFloat16 = 0xff80
	uvQNaN   BFloat16 = 0x7fc1
)

// Float32 bit layout (for conversion).
const (
	f32Exp   uint32 = 0x7f800000
	f32Round uint32 = 0x00007fff // rounding bias for truncation
	f32Shift uint32 = 16
)

// Frombits returns BFloat16 from raw bits. Frombits(Bits(x)) == x.
func Frombits(b uint16) BFloat16 { return BFloat16(b) }

// Fromfloat32 converts float32 to BFloat16 using round-to-nearest-even.
func Fromfloat32(f float32) BFloat16 {
	u := math.Float32bits(f)
	// Round to nearest even: add rounding bias + LSB for tie-breaking
	if u&f32Exp != f32Exp { // not Inf/NaN
		u += f32Round + ((u >> f32Shift) & 1)
	}
	return BFloat16(u >> f32Shift)
}

// NaN returns a quiet NaN.
func NaN() BFloat16 { return uvQNaN }

// Inf returns +Inf if sign >= 0, -Inf otherwise.
func Inf(sign int) BFloat16 {
	if sign < 0 {
		return uvNegInf
	}
	return uvPosInf
}

// Float32 converts to float32 (lossless).
func (f BFloat16) Float32() float32 { return math.Float32frombits(uint32(f) << f32Shift) }

// Float64 converts to float64 (lossless).
func (f BFloat16) Float64() float64 { return float64(f.Float32()) }

// Bits returns the raw bit representation.
func (f BFloat16) Bits() uint16 { return uint16(f) }

// Sign returns -1 if f < 0, 0 if f == 0 or NaN, +1 if f > 0.
func (f BFloat16) Sign() int {
	if f.IsNaN() || f&^signMask == 0 {
		return 0
	}
	if f&signMask != 0 {
		return -1
	}
	return 1
}

// String implements fmt.Stringer.
func (f BFloat16) String() string { return strconv.FormatFloat(float64(f.Float32()), 'f', -1, 32) }

// IsNaN reports whether f is NaN.
func (f BFloat16) IsNaN() bool { return f&expMask == expMask && f&coefMask != 0 }

// IsQuietNaN reports whether f is a quiet NaN.
func (f BFloat16) IsQuietNaN() bool { return f.IsNaN() && f&qnanBit != 0 }

// IsInf reports whether f is infinity (sign: 0=any, >0=+Inf, <0=-Inf).
func (f BFloat16) IsInf(sign int) bool {
	if sign == 0 {
		return f == uvPosInf || f == uvNegInf
	}
	return (sign > 0 && f == uvPosInf) || (sign < 0 && f == uvNegInf)
}

// IsFinite reports whether f is finite (not Inf or NaN).
func (f BFloat16) IsFinite() bool { return f&expMask != expMask }

// IsNormal reports whether f is normal (not zero, subnormal, Inf, or NaN).
func (f BFloat16) IsNormal() bool { e := f & expMask; return e != 0 && e != expMask }

// Signbit reports whether f is negative or negative zero.
func (f BFloat16) Signbit() bool { return f&signMask != 0 }
