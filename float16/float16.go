package float16

import (
	"errors"
	"math"
	"strconv"

	"github.com/gocnn/half"
)

// Compile-time interface compliance check.
var _ half.Float = Float16(0)

// Float16 represents IEEE 754 half-precision floating-point numbers (binary16).
type Float16 uint16

// Precision indicates the accuracy of float32 to Float16 conversion.
type Precision int

const (
	PrecisionExact     Precision = iota // exact, round-trips
	PrecisionUnknown                    // subnormal, may not round-trip
	PrecisionInexact                    // bits dropped, cannot round-trip
	PrecisionUnderflow                  // underflow
	PrecisionOverflow                   // overflow
)

// SmallestNonzero is the smallest positive nonzero Float16 (≈5.96e-08).
const SmallestNonzero = Float16(0x0001)

// ErrInvalidNaN is returned when input is not a valid NaN.
var ErrInvalidNaN = errors.New("float16: expected NaN input")

// Float16 bit layout.
const (
	signMask Float16 = 0x8000
	expMask  Float16 = 0x7c00
	coefMask Float16 = 0x03ff
	qnanBit  Float16 = 0x0200
	uvPosInf Float16 = 0x7c00
	uvNegInf Float16 = 0xfc00
	uvQNaN   Float16 = 0x7e01
	uvSNaN   Float16 = 0x7c01
)

// Float32 bit layout.
const (
	f32Sign   uint32 = 0x80000000
	f32Exp    uint32 = 0x7f800000
	f32Coef   uint32 = 0x007fffff
	f32Shift  uint32 = 23
	f32Bias   int32  = 127
	f32Hidden uint32 = 0x00800000
	f32Round  uint32 = 0x00001000
	f16Bias   int32  = 15
	f16ExpMax uint32 = 0x1f
)

// PrecisionFromfloat32 returns Precision without performing the conversion.
// Inf/NaN always report PrecisionExact. Designed for inlining (<0.5 ns/op).
func PrecisionFromfloat32(f32 float32) Precision {
	u := math.Float32bits(f32)
	if u == 0 || u == f32Sign {
		return PrecisionExact
	}

	exp := int32((u&f32Exp)>>f32Shift) - f32Bias
	coef := u & f32Coef

	switch {
	case exp == 128:
		return PrecisionExact
	case exp < -24:
		return PrecisionUnderflow
	case exp > 15:
		return PrecisionOverflow
	case coef&(f32Coef>>10) != 0:
		return PrecisionInexact
	case exp < -14:
		return PrecisionUnknown // subnormal
	default:
		return PrecisionExact
	}
}

// Frombits returns Float16 from IEEE 754 binary16 bits. Frombits(Bits(x)) == x.
func Frombits(b uint16) Float16 { return Float16(b) }

// Fromfloat32 converts float32 to Float16 using IEEE round-to-nearest-even.
func Fromfloat32(f float32) Float16 {
	return Float16(f32bitsToF16bits(math.Float32bits(f)))
}

// FromNaN32ps converts float32 NaN to Float16 NaN preserving sign and payload.
func FromNaN32ps(f float32) (Float16, error) {
	u := math.Float32bits(f)
	s, e, c := u&f32Sign, u&f32Exp, u&f32Coef
	if e != f32Exp || c == 0 {
		return uvSNaN, ErrInvalidNaN
	}
	f16 := Float16((s >> 16) | uint32(expMask) | (c >> 13))
	if f16&coefMask == 0 {
		f16 |= 1
	}
	return f16, nil
}

// NaN returns a quiet NaN.
func NaN() Float16 { return uvQNaN }

// Inf returns +Inf if sign >= 0, -Inf otherwise.
func Inf(sign int) Float16 {
	if sign < 0 {
		return uvNegInf
	}
	return uvPosInf
}

// Float32 converts to float32 (lossless).
func (f Float16) Float32() float32 { return math.Float32frombits(f16bitsToF32bits(uint16(f))) }

// Float64 converts to float64 (lossless).
func (f Float16) Float64() float64 { return float64(f.Float32()) }

// Bits returns the IEEE 754 binary16 representation.
func (f Float16) Bits() uint16 { return uint16(f) }

// Sign returns -1 if f < 0, 0 if f == 0 or NaN, +1 if f > 0.
func (f Float16) Sign() int {
	if f.IsNaN() || f&^signMask == 0 {
		return 0
	}
	if f&signMask != 0 {
		return -1
	}
	return 1
}

// String implements fmt.Stringer.
func (f Float16) String() string { return strconv.FormatFloat(float64(f.Float32()), 'f', -1, 32) }

// IsNaN reports whether f is NaN.
func (f Float16) IsNaN() bool { return f&expMask == expMask && f&coefMask != 0 }

// IsQuietNaN reports whether f is a quiet NaN.
func (f Float16) IsQuietNaN() bool { return f.IsNaN() && f&qnanBit != 0 }

// IsInf reports whether f is infinity (sign: 0=any, >0=+Inf, <0=-Inf).
func (f Float16) IsInf(sign int) bool {
	if sign == 0 {
		return f == uvPosInf || f == uvNegInf
	}
	return (sign > 0 && f == uvPosInf) || (sign < 0 && f == uvNegInf)
}

// IsFinite reports whether f is finite (not Inf or NaN).
func (f Float16) IsFinite() bool { return f&expMask != expMask }

// IsNormal reports whether f is normal (not zero, subnormal, Inf, or NaN).
func (f Float16) IsNormal() bool { e := f & expMask; return e != 0 && e != expMask }

// Signbit reports whether f is negative or negative zero.
func (f Float16) Signbit() bool { return f&signMask != 0 }

func f16bitsToF32bits(b uint16) uint32 {
	s := uint32(b>>15) << 31
	e := uint32((b >> 10) & 0x1f)
	m := uint32(b&0x03ff) << 13

	if e == f16ExpMax {
		if m == 0 {
			return s | f32Exp
		}
		return s | f32Exp | (1 << 22) | m
	}

	if e == 0 {
		if m == 0 {
			return s
		}
		e = 1
		for m&f32Exp == 0 {
			m <<= 1
			e--
		}
		m &= f32Coef
	}

	return s | (uint32(int32(e)+f32Bias-f16Bias) << 23) | m
}

func f32bitsToF16bits(b uint32) uint16 {
	s := b & f32Sign
	e := b & f32Exp
	m := b & f32Coef

	if e == f32Exp {
		q := uint32(0)
		if m != 0 {
			q = uint32(qnanBit)
		}
		return uint16((s >> 16) | uint32(expMask) | q | (m >> 13))
	}

	hs := s >> 16
	ue := int32(e>>f32Shift) - f32Bias
	he := ue + f16Bias

	if he >= int32(f16ExpMax) {
		return uint16(hs | uint32(expMask))
	}

	if he <= 0 {
		if 14-he > 24 {
			return uint16(hs)
		}
		c := m | f32Hidden
		hm := c >> uint32(14-he)
		rb := uint32(1) << uint32(13-he)
		if c&rb != 0 && c&(3*rb-1) != 0 {
			hm++
		}
		return uint16(hs | hm)
	}

	hex := uint32(he) << 10
	hm := m >> 13
	if m&f32Round != 0 && m&(3*f32Round-1) != 0 {
		return uint16((hs | hex | hm) + 1)
	}
	return uint16(hs | hex | hm)
}
