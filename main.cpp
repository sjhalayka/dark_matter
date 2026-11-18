// main.cpp
//
// Self-interacting gravitational field-line simulation.
// Four setups, selected by Params::mode (command line: --single / --two / --galaxy / --sun):
//   * SINGLE : one emitter at the origin
//   * SUN    : one emitter at the origin that is NOT a black hole: its radius is the Sun's
//              radius (4.3044e43 Planck lengths), about 2.36e5 times its own Schwarzschild
//              radius (1.8272e38 Planck lengths). The field lines are the chords themselves,
//              INSIDE the body: each line starts at a surface point, heads inward towards a
//              second random surface point and ends where it leaves the body. Random chords
//              fill a ball uniformly, so all of the line density sits inside the body and the
//              field outside it is the pure monopole g = M / r^2: Keplerian, v ~ r^-1/2.
//              Lengths are stored in a scaled unit (printed at start-up) because 4.3e43
//              does not fit in a float; the simulation itself is scale-free.
//              The exterior field is drawn as streamlines of g (key f): they start on the
//              body's surface and run outward against the pull to the edge of the box,
//              coloured by |g| on a log scale (red = strongest, at the surface; blue = 100x
//              weaker or less). The field is an instantaneous Poisson solve of the line
//              density, so it appears as the chords fill in rather than spreading at c.
//   * TWO    : two emitters on the x axis at +-D (the original setup)
//   * GALAXY : a galaxy of many emitters laid out as a thin spiral disk in the
//              x-z plane (exponential disk + logarithmic spiral arms + central bulge)
//
// Rotation curve (key r): around one emitter (the emitter in SINGLE mode, emitter 0 in
// TWO mode, the box centre in GALAXY mode) the field is sampled at a range of radii.
// At each radius a small axis-aligned box (half-size = one or two cells) is placed in
// several directions; the inward radial field of every grid cell inside the box is
// averaged. v(r) = sqrt(r * g_r(r)). A flat curve has log-log slope ~0, Keplerian -0.5.
// The table is printed and written to rotation_curve_step<N>.csv.
//
// Each field line has a fixed, deterministic origin and initial direction on its
// emitter's surface (regenerated from a hash of (emitter, line id)). A line is a
// CONNECTED CHAIN of nodes: node 0 sits at the origin and node k sits exactly one
// segment beyond node k-1, along node k's direction. Lines are therefore continuous
// by construction, however strongly they bend.
//
// Each step (dt = segment_length Planck times, signals move at c = 1):
//   1. Every node ages by one: old node j becomes node j+1, keeping its direction,
//      which is first turned by the gravity gradient at its old position. A fresh
//      node 0 with the line's initial direction is placed at the origin. So a bend
//      created at some point propagates outward along the line at c.
//   2. The chain is rebuilt outward from the origin: p[k] = p[k-1] + dir[k] * seg.
//      Each segment p[k-1] -> p[k] is deposited into a new density grid.
//   3. The chain is cut where it leaves the grid (lines pass through emitters).
//   4. The new grid and its gradient replace the old ones.
//
// The grid is a box with independent half-size and resolution per axis:
//   single / sun mode: Hx = margin_x * R,       Hy = margin_y * R,  Hz = margin_z * R
//   two-emitter mode:  Hx = D + margin_x * R,   Hy = margin_y * R,  Hz = margin_z * R
//   galaxy mode:       Hx = Rg + margin_x * R,  Hy = margin_y * R,  Hz = Rg + margin_z * R
//   (Rg = galaxy radius; the disk lies in the x-z plane, y is the thin axis)
//   cell size  cx = 2 Hx / Nx, etc. Cells may be non-cubic; the Poisson solver,
//   gradient and deposition all use the per-axis spacing.
//
// Emitter hit tests use a coarse uniform grid of emitters (cell = 2R), so the cost
// per node does not grow with the number of emitters.
//
// Storage is line-major: per line a uint32 node count followed by its nodes, in
// line-id order. If the estimate does not fit the RAM budget the store lives on
// disk as two swap files (read one, write the other, swap each step).
//
// Build (Linux):
//   g++ -O3 -march=native -std=c++17 main.cpp -o fieldlines -lGLEW -lglut -lGL -pthread
//
// Command line (a "triple" is either one value for all axes, or x,y,z):
//   --n <lines per emitter>   (default 1e4)
//   --grid <N> | <x,y,z>      (one value: N cells along the longest box axis, other axes
//                              proportional so cells are cubic; x,y,z: explicit counts. default 100)
//   --margin <triple>         (box size in emitter radii, default 6; see above)
//   --seg <length>            (segment length; <= 0 means the smallest cell size, which is
//                              also the default in sun mode)
//   --strength <k>            (self-gravity: max turn per step in radians, default 0.1; negative repels)
//   --iters <k>               (Poisson SOR iterations per step, default 10)
//   --model newton|classical|quantum   (emission direction model, default quantum)
//   --ram <GB>                (RAM budget for nodes before switching to disk, default 4)
//   --swap <dir>              (directory for disk swap files, default .)
//   --highlight <k>           (lines per emitter drawn at full opacity, default 16; 0 = off)
//   --single                  (one emitter at the origin)
//   --two                     (two emitters at +-D on the x axis)
//   --sun                     (one Sun-sized emitter at the origin, chords inside the body)
//   --cpu                    (do the grid work on the CPU instead of GPU compute shaders)
//   --end-at-emitters         (lines stop where they enter another emitter; default: pass through)
//  Galaxy mode:
//   --galaxy                  (galaxy of emitters; this is the default mode)
//   --emitters <k>            (number of emitters in the galaxy, default 100)
//   --galaxy-radius <r>       (disk radius in emitter radii, default 40)
//   --arms <k>                (number of spiral arms, default 2; 0 = no arms)
//   --thickness <t>           (disk vertical scatter, in emitter radii, default 1)
//
// Examples:
//   ./fieldlines --margin 20 --grid 200                 (bigger cubic box)
//   ./fieldlines --margin 30,15,15 --grid 400           (long box; grid becomes ~400 x 182 x 182)
//   ./fieldlines --galaxy --n 1000 --emitters 150 --grid 200 --highlight 2
//   ./fieldlines --single --margin 40 --grid 200        (single emitter; press r for its rotation curve)
//   ./fieldlines --sun --n 100000 --grid 128            (Sun; press r: slope should be close to -0.5)
//
// Controls: left-drag orbit, wheel / right-drag zoom, space pause, Esc quit,
//           c toggle colour (bend / emitter), [ ] bend colour sensitivity,
//           h toggle opaque highlighted lines, t toggle transparent (cloud) field lines,
//           v toggle density volume rendering, , / . volume brightness down / up,
//           p save large-format screenshot (PNG, window size x scale), + / - screenshot scale (default 4x),
//           f toggle the exterior gravitational-field streamlines (sun mode),
//           r compute the rotation curve (printed + CSV; runs between steps, also while paused),
//           m sun mode: mass from the interior chords (see Sim::computeMass).
//
// SUN-mode mass measurement (first-order observables):
//   The deposited line length is the mass: n straight chords of mean length (4/3) R carry one
//   solar mass, so G_eff = (Rs/2) / ((4/3) n R) converts grid units to geometric units
//   (G = c = 1). Along every chord two weak-field observables are integrated in double
//   precision from the Poisson-solved field of the chords themselves:
//     * deflection   delta = 2 * integral( g_perp dl )         (light bending, linear in M)
//     * Shapiro delay   dt = -2 * integral( phi dl )           (time delay, linear in M)
//   Each is divided by the same integral along the same chord for an analytic uniform ball of
//   one solar mass; the ratio is the measured mass in M_sun. These are NOT fixed by
//   construction: they depend on the field the chords actually produce.
//   The old estimate M = (3/4) <l> (unit-sphere mean chord) is still printed, but it is 1 for
//   ANY set of uniform straight chords, and the arc-length excess of bending is only second
//   order (~theta^2 ~ 1e-11 for the Sun), so it cannot measure the mass.
//   Bending: in SUN mode the turn per step is the physical one, 2 G_eff g_perp seg (~1e-7 rad),
//   unless --strength is given explicitly (then the old normalised, unphysical kick is used).
//   Note: node directions are stored as int16 octahedral (~3e-5 rad), so physical kicks this
//   small do not change the stored chords; the observables above are integrated separately.
//
// Colour: bend angle between consecutive segments of a line.
// Blue = straight ... red = most bent (red = 95th percentile of displayed bends).
// Emitter colour mode: orange (first emitter) ... blue (last emitter).
//
// GPU acceleration (OpenGL 4.3 compute shaders, default unless --cpu):
//   * Dirichlet boundary fill, red-black SOR Poisson solve, central-difference gradient,
//     max |grad| reduction (shared + global atomicMax) and the 3D density texture all
//     run on the GPU. phi stays resident on the GPU between steps (warm start).
//   * Per step only rho is uploaded and gx/gy/gz are read back (the CPU line pass needs them).
//   * The field-line pass (variable-length chains, optional disk streaming) stays on the
//     CPU threads. The GL context lives on the GUI thread, so the sim thread hands
//     the GPU solve to it and waits.

#define _CRT_SECURE_NO_WARNINGS

#include "GL/glew.h"
#include "GL/freeglut.h"
#pragma comment(lib, "freeglut")
#pragma comment(lib, "glew32")

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using std::size_t;

static const double PI = 3.14159265358979323846;

// ---------------------------------------------------------------- math

struct V3 { float x, y, z; };
static inline V3 operator+(V3 a, V3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
static inline V3 operator-(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static inline V3 operator*(V3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
static inline float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V3 cross(V3 a, V3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
static inline float len(V3 a) { return std::sqrt(dot(a, a)); }
static inline V3 norm(V3 a) { float l = len(a); return l > 0 ? a * (1.0f / l) : a; }

// ---------------------------------------------------------------- packed node (16 bytes)
// Position as 3 floats, direction (of the segment ending at this node) octahedral in two int16.

struct PackedNode { float p[3]; int16_t o[2]; };
static_assert(sizeof(PackedNode) == 16, "PackedNode must be 16 bytes");

static inline float signNZ(float v) { return v >= 0 ? 1.0f : -1.0f; }

static inline PackedNode packNode(V3 p, V3 v)
{
	float s = std::fabs(v.x) + std::fabs(v.y) + std::fabs(v.z);
	float x = v.x / s, y = v.y / s;
	if (v.z < 0) {
		float nx = (1.0f - std::fabs(y)) * signNZ(x);
		float ny = (1.0f - std::fabs(x)) * signNZ(y);
		x = nx; y = ny;
	}
	PackedNode n;
	n.p[0] = p.x; n.p[1] = p.y; n.p[2] = p.z;
	n.o[0] = (int16_t)std::lround(std::max(-1.0f, std::min(1.0f, x)) * 32767.0f);
	n.o[1] = (int16_t)std::lround(std::max(-1.0f, std::min(1.0f, y)) * 32767.0f);
	return n;
}

static inline void unpackNode(const PackedNode& n, V3& p, V3& v)
{
	p = { n.p[0], n.p[1], n.p[2] };
	float x = n.o[0] / 32767.0f, y = n.o[1] / 32767.0f;
	float z = 1.0f - std::fabs(x) - std::fabs(y);
	float t = std::max(-z, 0.0f);
	x += x >= 0 ? -t : t;
	y += y >= 0 ? -t : t;
	v = norm(V3{ x, y, z });
}

// ---------------------------------------------------------------- hashing RNG

static inline uint64_t splitmix(uint64_t x)
{
	x += 0x9E3779B97F4A7C15ull;
	x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
	x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
	return x ^ (x >> 31);
}
static inline double u01(uint64_t h) { return (h >> 11) * (1.0 / 9007199254740992.0); }

static inline V3 unitSphere(double u, double v)
{
	double z = 2.0 * u - 1.0, a = 2.0 * PI * v;
	double r = std::sqrt(std::max(0.0, 1.0 - z * z));
	return { (float)(r * std::cos(a)), (float)(r * std::sin(a)), (float)z };
}

// ---------------------------------------------------------------- parameters

enum Model { NEWTON, CLASSICAL, QUANTUM };
enum Mode { SINGLE, TWO, GALAXY, SUN };

// ---- the Sun in Planck units (SUN mode)
static const double PLANCK_LENGTH_M = 1.616255e-35;                 // CODATA 2018
static const double SUN_RADIUS_M = 6.957e8;                         // IAU nominal solar radius
static const double SUN_GM_SI = 1.3271244e20;                       // IAU nominal GM_sun, m^3 / s^2
static const double LIGHT_SPEED_SI = 299792458.0;
static const double SUN_RADIUS_PLANCK = SUN_RADIUS_M / PLANCK_LENGTH_M;                    // 4.3044e43
static const double SUN_RS_PLANCK = 2.0 * SUN_GM_SI / (LIGHT_SPEED_SI * LIGHT_SPEED_SI)
/ PLANCK_LENGTH_M;                                                                     // 1.8272e38
// Number of field lines of the real Sun: n = A_s / (4 ln 2) = pi Rs^2 / ln 2             // 1.5135e77
static const double SUN_LINES = PI * SUN_RS_PLANCK * SUN_RS_PLANCK / 0.69314718055994531;

struct Params {
	double n = 1e4;
	int N[3] = { 250, 250, 250 };              // cells per axis
	bool gridAuto = true;                      // one --grid value: cells along longest axis, others proportional
	double margin[3] = { 6.0, 6.0, 6.0 };      // box half-size in emitter radii (x: beyond D / Rg)
	double seg = 10;
	bool segSet = false;                       // --seg given (sun mode otherwise uses the cell size)
	double strength = seg / 100.0f;
	bool strengthSet = false;                  // --strength given (sun mode otherwise bends physically)
	int iters = 10;
	double ramGB = 24.0;
	size_t chunk = size_t(1) << 22;       // nodes per processing batch
	size_t maxDisplay = size_t(1) << 21;  // approx. segments sent to the display
	std::string swapDir = ".";
	Model model = QUANTUM;
	double highlight = 0.25;// 16;
	bool cpu = false;                     // force the CPU grid solver
	bool endAtEmitters = false;           // true: a line stops where it enters another emitter

	Mode mode = SINGLE;                   // SINGLE, TWO, GALAXY or SUN

	// ---- galaxy mode
	int emitters = 1000;                   // number of emitters in the galaxy
	double galaxyRadius = 40;             // disk radius in emitter radii
	int arms = 2;                         // spiral arms (0 = plain exponential disk)
	double thickness = 1.0;               // vertical scatter (sigma) in emitter radii
};

// Parse "v" or "x,y,z" into three doubles.
static void parseTriple(const std::string& s, double out[3], const char* opt)
{
	std::vector<double> v;
	std::stringstream ss(s);
	std::string item;
	while (std::getline(ss, item, ',')) v.push_back(std::stod(item));
	if (v.size() == 1) { out[0] = out[1] = out[2] = v[0]; return; }
	if (v.size() == 3) { out[0] = v[0]; out[1] = v[1]; out[2] = v[2]; return; }
	fprintf(stderr, "%s expects one value or x,y,z\n", opt);
	std::exit(1);
}

// Display vertex: one end of a line segment.
struct DispVert { float x, y, z, bend, em; };

// SUN mode: per-step statistics of the interior chords (summed over lines).
struct ChordStats {
	double lines = 0;      // lines processed
	double arcAll = 0;     // arc length of all lines (complete or still growing)
	double done = 0;       // lines that have reached the far surface (complete chords)
	double arc = 0;        // arc length of complete chords
	double span = 0;       // straight end-to-end distance of complete chords
	double aim = 0;        // straight chord the line was aimed along (2 R |s1 . d0|)
	double arc2 = 0;       // sum of (arc / R)^2, for the error of the mean
	// first-order observables of complete chords: x = measured (grid field), y = analytic
	// uniform ball of one solar mass; sums for the ratio and its delta-method error
	double dX = 0, dY = 0, dXX = 0, dYY = 0, dXY = 0;   // deflection (rad)
	double sX = 0, sY = 0, sXX = 0, sYY = 0, sXY = 0;   // Shapiro delay / R
	void add(const ChordStats& o) {
		lines += o.lines; arcAll += o.arcAll; done += o.done; arc += o.arc;
		span += o.span; aim += o.aim; arc2 += o.arc2;
		dX += o.dX; dY += o.dY; dXX += o.dXX; dYY += o.dYY; dXY += o.dXY;
		sX += o.sX; sY += o.sY; sXX += o.sXX; sYY += o.sYY; sXY += o.sXY;
	}
};

// ---------------------------------------------------------------- simulation

template <class F> static void parallelFor(size_t n, unsigned T, F f)
{
	std::vector<std::thread> th;
	for (unsigned t = 0; t < T; t++) {
		size_t a = n * t / T, b = n * (t + 1) / T;
		th.emplace_back(f, t, a, b);
	}
	for (auto& x : th) x.join();
}

class Sim {
public:
	Params P;
	unsigned T;
	int Nx, Ny, Nz;
	size_t NXY;                      // Nx * Ny (z stride)
	double Rcell = 1;                // radius used for layout / emitter lookup grid
	double R, D = 0, Rg = 0;         // emitter radius, half-separation (2-emitter), galaxy radius
	double unitPlanck = 1;           // Planck lengths per simulation length unit (SUN mode: >> 1)
	double Gsun = 0;                 // SUN mode: geometric G per unit of deposited line length
	double Mgeo = 0;                 // SUN mode: one solar mass in geometric sim units (= Rs / 2)
	bool physicalKick = false;       // SUN mode: bend by 2 G g_perp seg instead of strength / gRef
	double Hx, Hy, Hz, Hmax;         // box half-sizes
	double cx, cy, cz, cmin, cvol;   // cell sizes, smallest, volume
	double seg;
	std::vector<V3> centers;         // emitter centres
	uint32_t nE = 0;                 // number of emitters
	uint64_t lines;                  // lines per emitter
	uint64_t hlStride = 0;
	uint32_t maxChain = 0;   // safety cap on nodes per line

	// coarse emitter lookup grid (CSR): cell size egc, cells eg[0] x eg[1] x eg[2]
	double egc = 1;
	int eg[3] = { 1, 1, 1 };
	std::vector<uint32_t> egStart, egItems;

	std::vector<float> rho, phi, gx, gy, gz;
	float rhoRef = 0, gRef = 0;
	double massTotal = 0;                 // total deposited mass (for the boundary potential)
	ChordStats chord;                     // SUN mode: chord statistics of the last step
	std::function<void()> runGPU;         // if set: GPU grid solve (boundary, SOR, gradient)
	std::vector<std::vector<float>> tgrid;

	// line-major node store: [uint32 count][count * PackedNode] per line, in line order
	bool disk = false;
	std::vector<char> memCur, memNext;
	size_t memReadPos = 0;
	FILE* fin = nullptr;
	FILE* fout = nullptr;
	std::string files[2];
	int curFile = 0;

	uint64_t count = 0;
	uint64_t stepNo = 0;
	double lastStepSeconds = 0;

	std::mutex dispMtx;
	std::vector<DispVert> dispVerts;   // cloud segments, then highlighted segments
	size_t dispHlStart = 0;
	std::vector<DispVert> dispField;   // SUN mode: streamlines of g outside the body (bend = colour 0..1)
	std::vector<float> dispRho;
	float dispBendScale = 1.0f;
	float dispMeanBend = 0.0f;
	bool dispNew = false;

	// id = full line id (emitter * lines + line). When highlight < 1, the one
	// candidate line per emitter is kept only with probability `highlight`.
	bool isHighlighted(uint64_t id) const {
		if (!hlStride || ((id % lines) % hlStride) != 0) return false;
		if (P.highlight >= 1.0) return true;
		uint64_t h = splitmix((id / lines) ^ 0x9E3779B97F4A7C15ull);
		return u01(h) < P.highlight;
	}

	inline size_t ID(int i, int j, int k) const { return (size_t)i + (size_t)Nx * (size_t)j + NXY * (size_t)k; }

	const double separation = 20.5;

	explicit Sim(const Params& p) : P(p)
	{
		T = std::max(1u, std::thread::hardware_concurrency());
		lines = (uint64_t)P.n;
		hlStride = P.highlight > 0 ? std::max<uint64_t>(1, (uint64_t)(P.n / P.highlight)) : 0;
		R = std::sqrt(P.n * std::log(2.0) / PI);      // Schwarzschild radius of n field lines
		if (P.mode == SUN) {
			// The real Sun has SUN_LINES field lines; each simulated line stands for
			// SUN_LINES / n of them, which rescales lengths by sqrt(SUN_LINES / n). In that
			// unit the Schwarzschild radius is still sqrt(n ln2 / pi), and the body's radius is
			// SUN_RADIUS_PLANCK / SUN_RS_PLANCK (~2.36e5) times larger.
			unitPlanck = std::sqrt(SUN_LINES / P.n);
			R = SUN_RADIUS_PLANCK / unitPlanck;
		}
		Rcell = R;
		if (P.mode == GALAXY) {
			Rg = P.galaxyRadius * R;
			Hx = Rg + P.margin[0] * R;
			Hy = Hx;// P.margin[1] * R;
			Hz = Rg + P.margin[2] * R;
		}
		else if (P.mode == SINGLE || P.mode == SUN) {
			Hx = P.margin[0] * R;
			Hy = P.margin[1] * R;
			Hz = P.margin[2] * R;
		}
		else {
			D = separation * R;
			Hx = D + P.margin[0] * R;
			Hy = P.margin[1] * R;
			Hz = P.margin[2] * R;
		}
		if (Hx <= R || Hy <= R || Hz <= R) { fprintf(stderr, "--margin must exceed 1 (box must contain the emitters)\n"); std::exit(1); }
		Hmax = std::max(Hx, std::max(Hy, Hz));
		if (P.gridAuto) {
			// cubic cells: the longest axis gets P.N[0] cells, the others proportionally fewer
			double c = 2.0 * Hmax / P.N[0];
			Nx = (int)std::lround(2.0 * Hx / c);
			Ny = (int)std::lround(2.0 * Hy / c);
			Nz = (int)std::lround(2.0 * Hz / c);
		}
		else { Nx = P.N[0]; Ny = P.N[1]; Nz = P.N[2]; }
		Nx = std::max(4, Nx); Ny = std::max(4, Ny); Nz = std::max(4, Nz);
		NXY = (size_t)Nx * Ny;
		cx = 2.0 * Hx / Nx; cy = 2.0 * Hy / Ny; cz = 2.0 * Hz / Nz;
		cmin = std::min(cx, std::min(cy, cz));
		cvol = cx * cy * cz;
		seg = P.seg > 0 ? P.seg : cmin;
		if (P.mode == SUN && !P.segSet) seg = cmin;   // the Planck-scale default is meaningless here
		if (P.mode == SUN) {
			// n straight uniform chords deposit (4/3) n R of line length = one solar mass
			Mgeo = 0.5 * SUN_RS_PLANCK / unitPlanck;
			Gsun = Mgeo / ((4.0 / 3.0) * (double)lines * R);
			physicalKick = !P.strengthSet;
		}

		if (P.mode == GALAXY) buildGalaxy();
		else if (P.mode == SINGLE || P.mode == SUN) centers = { V3{ 0, 0, 0 } };
		else centers = { V3{ (float)-D, 0, 0 }, V3{ (float)D, 0, 0 } };
		nE = (uint32_t)centers.size();
		buildEmitterGrid();

		size_t cells = NXY * Nz;
		rho.assign(cells, 0); phi.assign(cells, 0); gx.assign(cells, 0); gy.assign(cells, 0); gz.assign(cells, 0);
		tgrid.assign(T, std::vector<float>(cells, 0));

		double pathSteps = 2.0 * Hmax / seg;
		maxChain = (uint32_t)std::min(4.0e9, 8.0 * pathSteps + 16);
		double totalLines = (double)nE * (double)lines;
		double estNodes = totalLines * pathSteps;
		double estBytes = estNodes * sizeof(PackedNode) + totalLines * 4;
		disk = estBytes > P.ramGB * 1073741824.0;
		files[0] = P.swapDir + "/fieldline_swap_0.bin";
		files[1] = P.swapDir + "/fieldline_swap_1.bin";
		if (disk) { std::remove(files[0].c_str()); std::remove(files[1].c_str()); }

		double gridGB = (5.0 + T) * cells * sizeof(float) / 1073741824.0;
		if (P.mode == GALAXY)
			printf("GALAXY mode: %u emitters, radius R = %.3f, galaxy radius %.3f, %d arms\n", nE, R, Rg, P.arms);
		else if (P.mode == SINGLE)
			printf("Single-emitter mode: radius R = %.3f at the origin\n", R);
		else if (P.mode == SUN) {
			printf("SUN mode: chords inside the body, lines end at its surface\n");
			printf("  Sun radius               = %.6e Planck lengths\n", SUN_RADIUS_PLANCK);
			printf("  Sun Schwarzschild radius = %.6e Planck lengths (R / Rs = %.4e)\n",
				SUN_RS_PLANCK, SUN_RADIUS_PLANCK / SUN_RS_PLANCK);
			printf("  Sun field lines          = %.6e (each simulated line stands for %.4e)\n",
				SUN_LINES, SUN_LINES / P.n);
			printf("  1 simulation length unit = %.6e Planck lengths; R = %.6e, Rs = %.4f in these units\n",
				unitPlanck, R, SUN_RS_PLANCK / unitPlanck);
			printf("  G_eff = %.6e (geometric mass per unit line length)\n", Gsun);
			if (physicalKick)
				printf("  bending: physical, 2 G_eff g_perp seg per step (~%.2e rad at the surface)\n",
					2.0 * Mgeo / (R * R) * seg);
			else
				printf("  bending: --strength %.4g rad per step at max |g| (NOT physical)\n", P.strength);
		}
		else
			printf("Two-emitter mode: radius R = %.3f, separation = %.3f\n", R, 2 * D);
		printf("Box half-size %.3f x %.3f x %.3f\n", Hx, Hy, Hz);
		printf("Grid %d x %d x %d (cell %.4f x %.4f x %.4f, %.2f GB of grids), segment length %.4f, %u threads\n",
			Nx, Ny, Nz, cx, cy, cz, gridGB, seg, T);
		printf("Lines per emitter %llu (%.3e total), est. steady-state nodes %.3e (%.2f GB) -> %s storage\n",
			(unsigned long long)lines, totalLines, estNodes, estBytes / 1073741824.0, disk ? "DISK" : "RAM");
		if (hlStride && P.highlight < 1.0)
			printf("Highlighting one line per emitter with probability %.3f (~%.1f lines total)\n",
				P.highlight, P.highlight * nE);
		else if (hlStride)
			printf("Highlighting every %llu-th line (~%llu per emitter)\n",
				(unsigned long long)hlStride, (unsigned long long)((lines + hlStride - 1) / hlStride));
		if (disk)
			printf("Disk mode needs up to ~%.1f GB free in '%s' (two swap files).\n",
				2.0 * estBytes / 1073741824.0, P.swapDir.c_str());
	}

	~Sim()
	{
		if (disk) { std::remove(files[0].c_str()); std::remove(files[1].c_str()); }
	}

	// Deterministic spiral-galaxy layout in the x-z plane. Emitters never overlap
	// (centre distance >= 2.5 R) and always lie fully inside the box.
	void buildGalaxy()
	{
		R = 0.00001;
		int want = std::max(1, P.emitters);
		double scale = Rg / 3.0;                 // exponential disk scale length
		double minD2 = (2.5 * R) * (2.5 * R);
		double pitch = 0.3;                      // spiral pitch angle (rad)
		double r0 = 2.0 * R;
		double xMax = Hx - 1.5 * R, yMax = Hy - 1.5 * R, zMax = Hz - 1.5 * R;
		uint64_t h = 0xC0FFEE1234ull;
		auto rnd = [&]() { h = splitmix(h); return u01(h); };
		auto gauss = [&]() {
			double u = std::max(1e-12, rnd()), v = rnd();
			return std::sqrt(-2.0 * std::log(u)) * std::cos(2.0 * PI * v);
			};

		long attempts = 0, maxAttempts = (long)want * 5000;
		while ((int)centers.size() < want && attempts++ < maxAttempts) {
			V3 c;
			if (rnd() < 0.15) {                  // bulge: flattened gaussian blob
				double s = Rg * 0.12;
				c = { (float)(gauss() * s), (float)(gauss() * s * 0.5), (float)(gauss() * s) };
			}
			else {                               // disk: truncated exponential radius
				double r = -scale * std::log(1.0 - rnd() * (1.0 - std::exp(-Rg / scale)));
				r = std::max(r, r0);
				double th;
				if (P.arms > 0) {
					int arm = std::min(P.arms - 1, (int)(rnd() * P.arms));
					th = 2.0 * PI * arm / P.arms + std::log(r / r0) / std::tan(pitch) + gauss() * 0.25;
				}
				else th = 2.0 * PI * rnd();
				c = { (float)(r * std::cos(th)), (float)(gauss() * P.thickness * R), (float)(r * std::sin(th)) };
			}
			if (std::fabs(c.x) > xMax || std::fabs(c.y) > yMax || std::fabs(c.z) > zMax) continue;
			bool ok = true;
			for (const V3& o : centers) { V3 d = c - o; if (dot(d, d) < minD2) { ok = false; break; } }
			if (ok) centers.push_back(c);
		}
		if ((int)centers.size() < want)
			fprintf(stderr, "Warning: only placed %zu of %d emitters (galaxy too crowded; raise --galaxy-radius)\n",
				centers.size(), want);
	}

	// Bin every emitter into each coarse cell its bounding box touches.
	void buildEmitterGrid()
	{
		egc = 2.0 * Rcell;             // lookup cell size (layout radius, not the emitter radius)
		eg[0] = std::max(1, (int)std::ceil(2.0 * Hx / egc));
		eg[1] = std::max(1, (int)std::ceil(2.0 * Hy / egc));
		eg[2] = std::max(1, (int)std::ceil(2.0 * Hz / egc));
		size_t ncell = (size_t)eg[0] * eg[1] * eg[2];
		std::vector<std::vector<uint32_t>> tmp(ncell);
		double H[3] = { Hx, Hy, Hz };
		for (uint32_t e = 0; e < nE; e++) {
			double c[3] = { centers[e].x, centers[e].y, centers[e].z };
			int lo[3], hi[3];
			for (int a = 0; a < 3; a++) {
				lo[a] = std::max(0, (int)std::floor((c[a] - R + H[a]) / egc));
				hi[a] = std::min(eg[a] - 1, (int)std::floor((c[a] + R + H[a]) / egc));
			}
			for (int k = lo[2]; k <= hi[2]; k++)
				for (int j = lo[1]; j <= hi[1]; j++)
					for (int i = lo[0]; i <= hi[0]; i++)
						tmp[(size_t)i + (size_t)eg[0] * ((size_t)j + (size_t)eg[1] * k)].push_back(e);
		}
		egStart.assign(ncell + 1, 0);
		egItems.clear();
		for (size_t q = 0; q < ncell; q++) {
			egStart[q] = (uint32_t)egItems.size();
			egItems.insert(egItems.end(), tmp[q].begin(), tmp[q].end());
		}
		egStart[ncell] = (uint32_t)egItems.size();
	}

	// Index of the emitter containing p, or -1.
	inline int emitterAt(V3 p) const
	{
		int i = (int)std::floor((p.x + Hx) / egc), j = (int)std::floor((p.y + Hy) / egc), k = (int)std::floor((p.z + Hz) / egc);
		if (i < 0 || j < 0 || k < 0 || i >= eg[0] || j >= eg[1] || k >= eg[2]) return -1;
		size_t q = (size_t)i + (size_t)eg[0] * ((size_t)j + (size_t)eg[1] * k);
		float r2 = (float)(R * R);
		for (uint32_t s = egStart[q]; s < egStart[q + 1]; s++) {
			uint32_t e = egItems[s];
			V3 d = p - centers[e];
			if (dot(d, d) < r2) return (int)e;
		}
		return -1;
	}

	void lineStart(uint32_t e, uint64_t i, V3& pos, V3& dir) const
	{
		uint64_t h1 = P.mode == GALAXY ? splitmix(splitmix((uint64_t)e + 0x51ED27ull) ^ i)
			: splitmix(((uint64_t)e << 62) ^ i);
		uint64_t h2 = splitmix(h1), h3 = splitmix(h2), h4 = splitmix(h3);
		V3 s1 = unitSphere(u01(h1), u01(h2));
		pos = centers[e] + s1 * (float)R;
		if (P.mode == SUN) {
			// Chord INSIDE the body: from surface point s1 inward towards a second uniformly
			// random surface point s2. (The other modes use the same chord, but follow its
			// continuation outward from s1.) Uniform random chords fill the ball uniformly.
			V3 s2 = unitSphere(u01(h3), u01(h4));
			V3 d = s2 - s1;
			dir = len(d) > 1e-6f ? norm(d) : s1 * -1.0f;
			return;
		}
		switch (P.model) {
		case NEWTON: dir = s1; break;
		case CLASSICAL: {
			V3 a = std::fabs(s1.x) < 0.9f ? V3{ 1, 0, 0 } : V3{ 0, 1, 0 };
			V3 uu = norm(cross(s1, a)), vv = cross(uu, s1);
			double r = std::sqrt(u01(h4)), ph = 2.0 * PI * u01(h3);
			float rx = (float)(r * std::cos(ph)), ry = (float)(r * std::sin(ph));
			float rz = (float)std::sqrt(std::max(0.0, 1.0 - u01(h4)));
			dir = norm(uu * rx + vv * ry + s1 * rz);
			break;
		}
		case QUANTUM: default: {
			V3 s2 = unitSphere(u01(h3), u01(h4));
			V3 d = s1 - s2;
			dir = len(d) > 1e-6f ? norm(d) : s1;
			break;
		}
		}
	}

	template <class F> inline void cic(V3 p, F f) const
	{
		double fx = (p.x + Hx) / cx - 0.5, fy = (p.y + Hy) / cy - 0.5, fz = (p.z + Hz) / cz - 0.5;
		int i0 = (int)std::floor(fx), j0 = (int)std::floor(fy), k0 = (int)std::floor(fz);
		float tx = (float)(fx - i0), ty = (float)(fy - j0), tz = (float)(fz - k0);
		for (int dz = 0; dz < 2; dz++) {
			int k = k0 + dz; if (k < 0 || k >= Nz) continue;
			float wz = dz ? tz : 1 - tz;
			for (int dy = 0; dy < 2; dy++) {
				int j = j0 + dy; if (j < 0 || j >= Ny) continue;
				float wy = dy ? ty : 1 - ty;
				for (int dx = 0; dx < 2; dx++) {
					int i = i0 + dx; if (i < 0 || i >= Nx) continue;
					float wx = dx ? tx : 1 - tx;
					f(ID(i, j, k), wx * wy * wz);
				}
			}
		}
	}

	inline void deposit(std::vector<float>& g, V3 a, V3 b) const
	{
		V3 d = b - a;
		float L = len(d);
		int ns = std::max(1, (int)std::ceil(L / (0.5 * cmin)));
		float w = (float)(L / ns / cvol);
		for (int s = 0; s < ns; s++) {
			V3 p = a + d * ((s + 0.5f) / ns);
			cic(p, [&](size_t idx, float wt) { g[idx] += w * wt; });
		}
	}

	inline V3 gradAt(V3 p) const
	{
		V3 g{ 0, 0, 0 };
		cic(p, [&](size_t idx, float wt) { g.x += gx[idx] * wt; g.y += gy[idx] * wt; g.z += gz[idx] * wt; });
		return g;
	}

	inline double phiAt(V3 p) const
	{
		double f = 0;
		cic(p, [&](size_t idx, float wt) { f += (double)phi[idx] * wt; });
		return f;
	}

	// Analytic uniform ball of one solar mass (geometric units), centred on emitter 0.
	inline void ballField(V3 p, double g[3], double& ph) const
	{
		double x = p.x - centers[0].x, y = p.y - centers[0].y, z = p.z - centers[0].z;
		double r2 = x * x + y * y + z * z, R3 = R * R * R;
		if (r2 <= R * R) {
			double s = -Mgeo / R3;
			g[0] = s * x; g[1] = s * y; g[2] = s * z;
			ph = -Mgeo * (3.0 * R * R - r2) / (2.0 * R3);
		}
		else {
			double r = std::sqrt(r2), s = -Mgeo / (r2 * r);
			g[0] = s * x; g[1] = s * y; g[2] = s * z;
			ph = -Mgeo / r;
		}
	}

	inline bool inside(V3 p) const
	{
		return p.x > -(float)Hx && p.x < (float)Hx && p.y > -(float)Hy && p.y < (float)Hy
			&& p.z > -(float)Hz && p.z < (float)Hz;
	}

	// Point where segment a -> b first enters emitter e (b is inside it).
	inline V3 emitterEntry(V3 a, V3 b, int e) const
	{
		V3 ab = b - a, ac = a - centers[e];
		float A = dot(ab, ab), B = 2 * dot(ab, ac), C = dot(ac, ac) - (float)(R * R);
		float disc = std::max(0.0f, B * B - 4 * A * C);
		float t = A > 0 ? (-B - std::sqrt(disc)) / (2 * A) : 0.0f;
		t = std::max(0.0f, std::min(1.0f, t));
		return a + ab * t;
	}

	// Point where segment a -> b leaves emitter e (a is inside it, b is not).
	inline V3 emitterExit(V3 a, V3 b, uint32_t e) const
	{
		double ab[3] = { (double)b.x - a.x, (double)b.y - a.y, (double)b.z - a.z };
		double ac[3] = { (double)a.x - centers[e].x, (double)a.y - centers[e].y, (double)a.z - centers[e].z };
		double A = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
		double B = 2.0 * (ab[0] * ac[0] + ab[1] * ac[1] + ab[2] * ac[2]);
		double C = ac[0] * ac[0] + ac[1] * ac[1] + ac[2] * ac[2] - R * R;
		double disc = std::max(0.0, B * B - 4.0 * A * C);
		double t = A > 0 ? (-B + std::sqrt(disc)) / (2.0 * A) : 0.0;
		t = std::max(0.0, std::min(1.0, t));
		return a + (b - a) * (float)t;
	}

	void computeGradient()
	{
		if (runGPU) runGPU();
		else computeGradientCPU();
	}

	void computeGradientCPU()
	{
		const double M = massTotal;

		// Dirichlet boundary: monopole potential of the total mass
		for (int k = 0; k < Nz; k++) for (int j = 0; j < Ny; j++) for (int i = 0; i < Nx; i++) {
			if (i > 0 && i < Nx - 1 && j > 0 && j < Ny - 1 && k > 0 && k < Nz - 1) continue;
			double x = -Hx + (i + 0.5) * cx, y = -Hy + (j + 0.5) * cy, z = -Hz + (k + 0.5) * cz;
			phi[ID(i, j, k)] = (float)(-M / std::sqrt(x * x + y * y + z * z));
		}

		// Red-black SOR for  lap(phi) = 4 pi rho  on an anisotropic grid
		const float ax = (float)(1.0 / (cx * cx)), ay = (float)(1.0 / (cy * cy)), az = (float)(1.0 / (cz * cz));
		const float invDiag = 1.0f / (2.0f * (ax + ay + az));
		const float fourPi = (float)(4.0 * PI), w = 1.85f;
		const size_t sx = 1, sy = (size_t)Nx, sz = NXY;
		for (int it = 0; it < P.iters; it++)
			for (int color = 0; color < 2; color++)
				parallelFor((size_t)(Nz - 2), T, [&](unsigned, size_t a, size_t b) {
				for (size_t kk = a; kk < b; kk++) {
					int k = (int)kk + 1;
					for (int j = 1; j < Ny - 1; j++) {
						int i0 = 1 + ((j + k + color + 1) & 1);
						for (int i = i0; i < Nx - 1; i += 2) {
							size_t c = ID(i, j, k);
							float nb = ax * (phi[c - sx] + phi[c + sx])
								+ ay * (phi[c - sy] + phi[c + sy])
								+ az * (phi[c - sz] + phi[c + sz]);
							float gs = (nb - fourPi * rho[c]) * invDiag;
							phi[c] += w * (gs - phi[c]);
						}
					}
				}
					});

		const float ix = (float)(1.0 / cx), iy = (float)(1.0 / cy), iz = (float)(1.0 / cz);
		auto at = [&](int i, int j, int k) { return phi[ID(i, j, k)]; };
		std::vector<float> tmax(T, 0.0f);
		parallelFor((size_t)Nz, T, [&](unsigned t, size_t a, size_t b) {
			for (size_t kk = a; kk < b; kk++) {
				int k = (int)kk;
				for (int j = 0; j < Ny; j++)
					for (int i = 0; i < Nx; i++) {
						size_t idx = ID(i, j, k);
						int ip = std::min(i + 1, Nx - 1), im = std::max(i - 1, 0);
						int jp = std::min(j + 1, Ny - 1), jm = std::max(j - 1, 0);
						int kp = std::min(k + 1, Nz - 1), km = std::max(k - 1, 0);
						gx[idx] = -(at(ip, j, k) - at(im, j, k)) * ix / (float)(ip - im);
						gy[idx] = -(at(i, jp, k) - at(i, jm, k)) * iy / (float)(jp - jm);
						gz[idx] = -(at(i, j, kp) - at(i, j, km)) * iz / (float)(kp - km);
						float m = gx[idx] * gx[idx] + gy[idx] * gy[idx] + gz[idx] * gz[idx];
						tmax[t] = std::max(tmax[t], m);
					}
			}
			});
		gRef = 0;
		for (float m : tmax) gRef = std::max(gRef, std::sqrt(m));
	}

	// SUN mode: streamlines of the gravitational field outside the body, for display only.
	// Seeds are spread evenly over the surface (Fibonacci sphere); each streamline steps
	// outward against g until it leaves the grid. The colour value stored in `bend` is
	// 1 + log10(|g| / gRef) / 2, clamped to 0..1 (two decades below the grid maximum).
	void buildFieldLines(std::vector<DispVert>& out, int seeds = 600) const
	{
		if (gRef <= 0) return;
		const float h = (float)cmin;
		const int maxSteps = (int)(4.0 * Hmax / cmin) + 8;
		const double golden = PI * (3.0 - std::sqrt(5.0));
		for (int k = 0; k < seeds; k++) {
			double y = 1.0 - 2.0 * (k + 0.5) / seeds, rr = std::sqrt(std::max(0.0, 1.0 - y * y)), a = golden * k;
			V3 s = { (float)(rr * std::cos(a)), (float)y, (float)(rr * std::sin(a)) };
			V3 p = centers[0] + s * (float)R, prevStep = s;
			for (int n = 0; n < maxSteps; n++) {
				V3 g = gradAt(p);
				float m = len(g);
				if (!(m > 0)) break;
				V3 q = p - g * (h / m);             // g pulls inward; walk outward
				if (!inside(q)) break;
				if (dot(q - p, prevStep) <= 0) break;   // doubled back (unsettled field in the first steps)
				prevStep = q - p;
				float t = 1.0f + 0.5f * std::log10(m / gRef);
				addSeg(out, p, q, std::max(0.0f, std::min(1.0f, t)), 0.0f);
				p = q;
			}
		}
	}

	// ---- node storage streams (byte streams)
	void openStreams()
	{
		if (disk) {
			fin = stepNo > 0 ? std::fopen(files[curFile].c_str(), "rb") : nullptr;
			fout = std::fopen(files[curFile ^ 1].c_str(), "wb");
			if (!fout) { fprintf(stderr, "Cannot open swap file %s\n", files[curFile ^ 1].c_str()); std::exit(1); }
			std::setvbuf(fout, nullptr, _IOFBF, 1 << 24);
			if (fin) std::setvbuf(fin, nullptr, _IOFBF, 1 << 24);
		}
		else {
			memReadPos = 0;
			memNext.clear();
			memNext.reserve(memCur.size() + (size_t)nE * lines * (4 + 2 * sizeof(PackedNode)));
		}
	}
	bool readBytes(void* dst, size_t n)
	{
		if (n == 0) return true;
		if (disk) return fin && std::fread(dst, 1, n, fin) == n;
		if (memReadPos + n > memCur.size()) return false;
		std::memcpy(dst, memCur.data() + memReadPos, n);
		memReadPos += n;
		return true;
	}
	void writeBytes(const std::vector<char>& v)
	{
		if (v.empty()) return;
		if (disk) {
			if (std::fwrite(v.data(), 1, v.size(), fout) != v.size()) {
				fprintf(stderr, "Swap file write failed (disk full?)\n");
				std::exit(1);
			}
		}
		else memNext.insert(memNext.end(), v.begin(), v.end());
	}
	void closeStreams()
	{
		if (disk) {
			if (fin) std::fclose(fin);
			std::fclose(fout);
			fin = fout = nullptr;
			curFile ^= 1;
		}
		else {
			std::swap(memCur, memNext);
			memNext.clear();
			memNext.shrink_to_fit();
		}
	}

	static inline void addSeg(std::vector<DispVert>& dv, V3 a, V3 b, float bend, float em)
	{
		dv.push_back({ a.x, a.y, a.z, bend, em });
		dv.push_back({ b.x, b.y, b.z, bend, em });
	}

	// Age line `id` by one step and rebuild it as a connected chain from its origin.
	// in[0..m) are the line's old nodes, ordered from the origin outward.
	uint64_t processLine(uint64_t id, const PackedNode* in, uint32_t m, float kick,
		std::vector<char>& out, std::vector<float>& g, std::vector<DispVert>* dv, ChordStats* cs = nullptr) const
	{
		uint32_t e = (uint32_t)(id / lines);
		float em = nE > 1 ? (float)e / (float)(nE - 1) : 0.0f, fseg = (float)seg;
		V3 prev, d0;
		lineStart(e, id % lines, prev, d0);
		const V3 start = prev;
		double arc = 0;
		if (cs) cs->lines++;
		struct Flush {   // arc of every line, complete or not
			ChordStats* cs; double& arc;
			~Flush() { if (cs) cs->arcAll += arc; }
		} flush{ cs, arc };

		size_t hdr = out.size();
		out.resize(hdr + 4);
		uint32_t cnt = 0;
		auto push = [&](V3 q, V3 v) {
			PackedNode pn = packNode(q, v);
			size_t s = out.size();
			out.resize(s + sizeof(PackedNode));
			std::memcpy(out.data() + s, &pn, sizeof(PackedNode));
			cnt++;
			};

		push(prev, d0);                     // node 0 at the origin
		V3 prevDir = d0;

		// SUN mode: first-order observables along the chord (double precision, midpoint rule)
		double defG[3] = { 0, 0, 0 }, defA[3] = { 0, 0, 0 }, shG = 0, shA = 0;
		auto observe = [&](V3 a, V3 b, V3 v) {
			if (!cs) return;
			double L = len(b - a);
			V3 m = a + (b - a) * 0.5f;
			V3 g = gradAt(m);
			double gv = (double)g.x * v.x + (double)g.y * v.y + (double)g.z * v.z;
			double ga[3], pa;
			ballField(m, ga, pa);
			double gav = ga[0] * v.x + ga[1] * v.y + ga[2] * v.z;
			double vv[3] = { v.x, v.y, v.z }, gg[3] = { g.x, g.y, g.z };
			for (int a3 = 0; a3 < 3; a3++) {
				defG[a3] += 2.0 * Gsun * (gg[a3] - gv * vv[a3]) * L;
				defA[a3] += 2.0 * (ga[a3] - gav * vv[a3]) * L;
			}
			shG += -2.0 * Gsun * phiAt(m) * L / R;
			shA += -2.0 * pa * L / R;
			};
		uint32_t lim = std::min<uint32_t>(m, maxChain - 1);
		for (uint32_t j = 0; j < lim; j++) {
			V3 op, v;
			unpackNode(in[j], op, v);       // old node j -> new node j+1
			if (kick != 0) v = norm(v + gradAt(op) * kick);
			V3 np = prev + v * fseg;        // attached to its inner neighbour: continuous
			float bend = std::acos(std::max(-1.0f, std::min(1.0f, dot(v, prevDir))));
			if (!inside(np)) break;         // line ends: left the grid
			if (P.mode == SUN) {            // chords live inside the body: end at its surface
				V3 dc = np - centers[e];
				if (dot(dc, dc) >= (float)(R * R)) {
					V3 hit = emitterExit(prev, np, e);
					deposit(g, prev, hit);
					if (dv) addSeg(*dv, prev, hit, bend, em);
					push(hit, v);
					arc += len(hit - prev);
					observe(prev, hit, v);
					if (cs) {
						// measured deflection projected on the analytic one (sign-safe), and its size
						double nA = std::sqrt(defA[0] * defA[0] + defA[1] * defA[1] + defA[2] * defA[2]);
						double x = nA > 0 ? (defG[0] * defA[0] + defG[1] * defA[1] + defG[2] * defA[2]) / nA : 0.0;
						cs->dX += x; cs->dY += nA; cs->dXX += x * x; cs->dYY += nA * nA; cs->dXY += x * nA;
						cs->sX += shG; cs->sY += shA; cs->sXX += shG * shG; cs->sYY += shA * shA; cs->sXY += shG * shA;
						V3 s1 = (start - centers[e]) * (float)(1.0 / R);
						double a = arc / R;
						cs->done++;
						cs->arc += arc;
						cs->arc2 += a * a;
						cs->span += len(hit - start);
						cs->aim += 2.0 * R * std::fabs(dot(s1, d0));
					}
					break;
				}
			}
			if (P.endAtEmitters) {
				int h = emitterAt(np);
				if (h >= 0 && (uint32_t)h != e) {        // entered a different emitter
					V3 hit = emitterEntry(prev, np, h);
					deposit(g, prev, hit);
					if (dv) addSeg(*dv, prev, hit, bend, em);
					push(hit, v);
					break;                               // line ends at the emitter's surface
				}
			}
			// Otherwise entering an emitter does not end the line: it passes straight through.
			observe(prev, np, v);
			deposit(g, prev, np);
			if (dv) addSeg(*dv, prev, np, bend, em);
			push(np, v);
			arc += fseg;
			prev = np; prevDir = v;
		}
		std::memcpy(out.data() + hdr, &cnt, 4);
		return cnt;
	}

	void step()
	{
		auto t0 = std::chrono::steady_clock::now();

		parallelFor(T, T, [&](unsigned, size_t a, size_t b) {
			for (size_t t = a; t < b; t++) std::fill(tgrid[t].begin(), tgrid[t].end(), 0.0f);
			});

		openStreams();

		// display: whole lines are selected so they can be drawn as connected chains
		uint64_t dstride = std::max<uint64_t>(1, count / P.maxDisplay);
		auto showLine = [&](uint64_t id) { return splitmix(id ^ 0x5DEECE66Dull) % dstride == 0; };

		float kick = physicalKick ? (float)(2.0 * Gsun * seg)          // turn = 2 G g_perp seg
			: gRef > 0 ? (float)(P.strength / gRef) : 0.0f;
		bool haveInput = stepNo > 0;

		std::vector<std::vector<char>> outs(T);
		std::vector<std::vector<DispVert>> dCloud(T), dHl(T);
		std::vector<uint64_t> tcount(T, 0);
		std::vector<ChordStats> tchord(T);
		std::vector<PackedNode> inBuf;
		std::vector<size_t> off;
		std::vector<uint32_t> cnts;

		uint64_t total = (uint64_t)nE * lines;
		for (uint64_t L0 = 0; L0 < total; ) {
			inBuf.clear(); off.clear(); cnts.clear();
			uint64_t L1 = L0;
			while (L1 < total && (L1 == L0 || inBuf.size() < P.chunk) && (L1 - L0) < P.chunk) {
				uint32_t c = 0;
				if (haveInput && !readBytes(&c, 4)) {
					fprintf(stderr, "Node store read failed\n"); std::exit(1);
				}
				off.push_back(inBuf.size());
				cnts.push_back(c);
				if (c) {
					size_t s = inBuf.size();
					inBuf.resize(s + c);
					if (!readBytes(inBuf.data() + s, (size_t)c * sizeof(PackedNode))) {
						fprintf(stderr, "Node store read failed\n"); std::exit(1);
					}
				}
				L1++;
			}

			parallelFor((size_t)(L1 - L0), T, [&](unsigned t, size_t a, size_t b) {
				for (size_t i = a; i < b; i++) {
					uint64_t id = L0 + i;
					std::vector<DispVert>* dv = nullptr;
					if (isHighlighted(id)) dv = &dHl[t];
					else if (showLine(id)) dv = &dCloud[t];
					tcount[t] += processLine(id, inBuf.data() + off[i], cnts[i], kick, outs[t], tgrid[t], dv,
						P.mode == SUN ? &tchord[t] : nullptr);
				}
				});
			for (unsigned t = 0; t < T; t++) { writeBytes(outs[t]); outs[t].clear(); }
			L0 = L1;
		}

		closeStreams();
		count = 0;
		for (auto c : tcount) count += c;
		chord = ChordStats();
		for (auto& c : tchord) chord.add(c);

		size_t cells = rho.size();
		std::vector<double> tM(T, 0.0);
		std::vector<float> tR(T, 0.0f);
		parallelFor(cells, T, [&](unsigned t, size_t a, size_t b) {
			double m = 0; float mx = 0;
			for (size_t i = a; i < b; i++) {
				float s = 0;
				for (unsigned u = 0; u < T; u++) s += tgrid[u][i];
				rho[i] = s;
				m += s; mx = std::max(mx, s);
			}
			tM[t] = m; tR[t] = mx;
			});
		massTotal = 0; rhoRef = 0;
		for (unsigned t = 0; t < T; t++) { massTotal += tM[t]; rhoRef = std::max(rhoRef, tR[t]); }
		massTotal *= cvol;
		// The boundary potential uses half the mass in the other modes (kept as it was). In SUN
		// mode the Shapiro delay needs the absolute potential, so the boundary must carry the
		// full mass.
		if (P.mode != SUN) massTotal *= 0.5;

		computeGradient();
		stepNo++;

		std::vector<DispVert> field;
		if (P.mode == SUN) buildFieldLines(field);

		// assemble display buffer: cloud first, highlighted lines last
		std::vector<DispVert> verts;
		size_t nv = 0;
		for (unsigned t = 0; t < T; t++) nv += dCloud[t].size() + dHl[t].size();
		verts.reserve(nv);
		for (auto& d : dCloud) verts.insert(verts.end(), d.begin(), d.end());
		size_t hlStart = verts.size();
		for (auto& d : dHl) verts.insert(verts.end(), d.begin(), d.end());

		float scale = dispBendScale, mean = 0;
		if (!verts.empty()) {
			std::vector<float> b;
			size_t nseg = verts.size() / 2;
			size_t sstep = std::max<size_t>(1, nseg / 65536);
			double s = 0;
			for (size_t i = 0; i < nseg; i++) s += verts[2 * i].bend;
			mean = (float)(s / nseg);
			for (size_t i = 0; i < nseg; i += sstep) b.push_back(verts[2 * i].bend);
			size_t q = b.size() * 95 / 100;
			std::nth_element(b.begin(), b.begin() + q, b.end());
			scale = std::max(b[q], 1e-4f);
		}

		{
			std::lock_guard<std::mutex> lock(dispMtx);
			dispVerts.swap(verts);
			dispField.swap(field);
			dispHlStart = hlStart;
			dispBendScale = scale;
			dispMeanBend = mean;
			if (!runGPU) {                 // GPU mode writes the density texture directly
				dispRho.resize(cells);
				float inv = rhoRef > 0 ? 1.0f / rhoRef : 0.0f;
				for (size_t i = 0; i < cells; i++) dispRho[i] = rho[i] * inv;
			}
			dispNew = true;
		}

		lastStepSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
	}

	// SUN mode: mass measured from the interior chords (key m).
	//
	// Each line is a chord of the body. Rescale every chord by 1/R, i.e. place it on a UNIT
	// sphere; its length is then l. For straight chords between two uniform random surface
	// points, <l> = 4/3 exactly (chord length 2 sin(theta/2), theta uniform on the sphere).
	//
	// Now place the same chords on a SCHWARZSCHILD-sized sphere, radius Rs = 2M (G = c = 1):
	// the mean chord there is <L> = l Rs = (4/3) Rs = (8/3) M. So
	//     M_chord = (3/8) <L>_Rs = (3/8) <l> Rs,      M_chord / M_sun = (3/4) <l>
	// which is exactly 1 solar mass for straight chords. CAUTION: this holds for ANY set of
	// uniform straight chords, whatever the mass, so it is a normalisation, not a measurement;
	// and bending changes arc/span only at second order (theta^2), ~1e-11 for the Sun.
	//
	// The real measurement uses first-order observables integrated along each chord from the
	// chords' own Poisson field: light deflection 2 int g_perp dl and Shapiro delay
	// -2 int phi dl, each divided by the same integral for an analytic uniform ball of one
	// solar mass. Ratio = M / M_sun.
	//
	// For comparison the exterior field gives a Gauss's-law mass, M_gauss = <r^2 g_r> over
	// spheres outside the body, normalised the same way (the deposited "mass" of a full set of
	// straight chords, (4/3) n R in simulation units, is one solar mass).
	void computeMass()
	{
		if (P.mode != SUN) { printf("Mass measure: SUN mode only\n"); return; }
		if (stepNo == 0) { printf("Mass measure: no chords yet (run at least one step)\n"); return; }
		const ChordStats& c = chord;
		const double Rs_sim = SUN_RS_PLANCK / unitPlanck;
		const double PLANCK_MASS_KG = 2.176434e-8;
		const double MSUN_PLANCK = SUN_RS_PLANCK / 2.0;

		printf("\nMass from interior chords, step %llu\n", (unsigned long long)stepNo);
		printf("  lines %.0f, complete chords %.0f (%.2f%%)%s\n", c.lines, c.done,
			c.lines > 0 ? 100.0 * c.done / c.lines : 0.0,
			c.done < c.lines ? "  -- still filling; wait ~2R/seg steps for all chords" : "");
		if (c.done < 1) return;

		double l = c.arc / c.done / R;                         // mean arc on the unit sphere
		double var = std::max(0.0, c.arc2 / c.done - l * l);
		double el = std::sqrt(var / c.done);                    // error of the mean
		double ls = c.span / c.done / R, la = c.aim / c.done / R;
		printf("  unit sphere:   <arc> = %.6f +- %.6f   <span> = %.6f   <aimed chord> = %.6f   (straight: 4/3 = %.6f)\n",
			l, el, ls, la, 4.0 / 3.0);
		printf("                 bending excess arc/span - 1 = %+.4e\n", ls > 0 ? l / ls - 1.0 : 0.0);

		double M = 0.75 * l, eM = 0.75 * el;                    // in solar masses
		printf("  Schwarzschild sphere (Rs = %.6e Planck, %.4f sim units):\n", SUN_RS_PLANCK, Rs_sim);
		printf("                 <L>_Rs = %.6e Planck lengths   ->  M_chord = (3/8)<L> = %.6e Planck masses\n",
			l * SUN_RS_PLANCK, 0.375 * l * SUN_RS_PLANCK);
		printf("                 M_chord = %.6f +- %.6f M_sun = %.6e kg   (span only: %.6f M_sun)\n",
			M, eM, M * MSUN_PLANCK * PLANCK_MASS_KG, 0.75 * ls);

		// all lines, incomplete ones included: total chord length on the unit sphere per line
		double lAll = c.arcAll / c.lines / R;
		printf("  all lines:     sum(l) / n = %.6f  ->  %.6f M_sun (equals M_chord once every chord is complete)\n",
			lAll, 0.75 * lAll);
		printf("                 (M_chord = 1 for any uniform straight chords: a normalisation, not a measurement)\n");

		// first-order observables: ratio of sums X / Y with delta-method error
		auto ratio = [&](double X, double Y, double XX, double YY, double XY, double& err) {
			double n = c.done, r = Y != 0 ? X / Y : 0.0;
			double mx = X / n, my = Y / n;
			double vx = XX / n - mx * mx, vy = YY / n - my * my, cxy = XY / n - mx * my;
			double v = (vx - 2.0 * r * cxy + r * r * vy) / (my * my * n);
			err = std::sqrt(std::max(0.0, v));
			return r;
			};
		double eD, eS;
		double MD = ratio(c.dX, c.dY, c.dXX, c.dYY, c.dXY, eD);
		double MS = ratio(c.sX, c.sY, c.sXX, c.sYY, c.sXY, eS);
		printf("  light deflection 2 int g_perp dl:  mean %.6e rad (1 M_sun ball: %.6e)  ->  M = %.6f +- %.6f M_sun\n",
			c.dX / c.done, c.dY / c.done, MD, eD);
		printf("  Shapiro delay   -2 int phi dl / R: mean %.6e     (1 M_sun ball: %.6e)  ->  M = %.6f +- %.6f M_sun\n",
			c.sX / c.done, c.sY / c.done, MS, eS);
		printf("                 (these depend on the field the chords produce; grid discretisation biases\n"
			"                  them at the level of a few cells / R)\n");

		// Gauss's law on spheres outside the body
		double Hmin = std::min(Hx, std::min(Hy, Hz));
		double r0 = R + 2.0 * cmin, r1 = 0.9 * Hmin - 2.0 * cmin;
		double Mnorm = (4.0 / 3.0) * (double)lines * R;          // one solar mass in sim units
		if (r1 > r0) {
			printf("  Gauss's law:   %10s %10s %14s %12s\n", "r/R", "r/Rs", "<r^2 g_r>/M0", "M_sun");
			const int seeds = 400; const double golden = PI * (3.0 - std::sqrt(5.0));
			for (int n = 0; n < 6; n++) {
				double r = r0 + (r1 - r0) * n / 5.0, s = 0;
				for (int k = 0; k < seeds; k++) {
					double y = 1.0 - 2.0 * (k + 0.5) / seeds, rr = std::sqrt(std::max(0.0, 1.0 - y * y)), a = golden * k;
					V3 d = { (float)(rr * std::cos(a)), (float)y, (float)(rr * std::sin(a)) };
					V3 g = gradAt(centers[0] + d * (float)r);
					s += -dot(g, d) * r * r;
				}
				s /= seeds;
				printf("                 %10.4f %10.2f %14.6f %12.6f\n", r / R, r / Rs_sim,
					massTotal > 0 ? s / massTotal : 0.0, s / Mnorm);
			}
			printf("                 (column 3 = 1 means the Poisson solve has converged to the monopole)\n");
		}
		printf("  deposited line mass = %.6f M_sun\n\n", massTotal / Mnorm);
	}

	// Rotation curve around a single emitter, sampled with small axis-aligned boxes.
	// Must run while no step() is in progress (the sim thread calls it between steps),
	// so gx/gy/gz are stable.
	void computeRotationCurve(int nRadii = 40)
	{
		if (stepNo == 0) { printf("Rotation curve: no field yet (run at least one step)\n"); return; }

		V3 c = P.mode == GALAXY ? V3{ 0, 0, 0 } : centers[0];
		const char* what = P.mode == GALAXY ? "box centre" : (P.mode == TWO ? "emitter 0" : P.mode == SUN ? "the Sun" : "the emitter");

		// small AABB: half-size of ~1.5 cells along each axis (at least 3 cells wide)
		const double bx = 1.5 * cx, by = 1.5 * cy, bz = 1.5 * cz;
		const double bmax = std::max(bx, std::max(by, bz));

		// sampling directions: 6 axes + 8 cube diagonals (galaxy: 8 in-plane x-z directions)
		std::vector<V3> dirs;
		if (P.mode == GALAXY) {
			for (int a = 0; a < 8; a++) {
				double t = 2.0 * PI * a / 8;
				dirs.push_back({ (float)std::cos(t), 0, (float)std::sin(t) });
			}
		}
		else {
			dirs = { {1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1} };
			for (int s = 0; s < 8; s++)
				dirs.push_back(norm(V3{ (s & 1) ? 1.f : -1.f, (s & 2) ? 1.f : -1.f, (s & 4) ? 1.f : -1.f }));
		}

		// radius range: just outside the emitter (plus the box) to the farthest radius at
		// which at least one box still fits inside the grid
		double rMin = (P.mode == GALAXY ? 0.0 : R) + 1.5 * bmax;
		double rMax = 0;
		for (V3 d : dirs) {
			double lim = 1e30;
			double cc[3] = { c.x, c.y, c.z }, dd[3] = { d.x, d.y, d.z }, H[3] = { Hx, Hy, Hz }, b[3] = { bx, by, bz };
			for (int a = 0; a < 3; a++)
				if (std::fabs(dd[a]) > 1e-6) {
					double edge = dd[a] > 0 ? H[a] - b[a] - cc[a] : H[a] - b[a] + cc[a];
					lim = std::min(lim, edge / std::fabs(dd[a]));
				}
			rMax = std::max(rMax, lim);
		}
		rMax *= 0.98;
		if (rMax <= rMin) { printf("Rotation curve: box too small around %s\n", what); return; }

		printf("\nRotation curve around %s at (%.3f, %.3f, %.3f), step %llu\n", what, c.x, c.y, c.z, (unsigned long long)stepNo);
		printf("  sample box half-size %.4f x %.4f x %.4f, %zu directions, r = %.3f .. %.3f\n", bx, by, bz, dirs.size(), rMin, rMax);
		printf("  %12s %8s %14s %14s %12s %12s %6s\n", "r", "r/R", "g_r (inward)", "|g|", "v", "v_kepler", "cells");

		std::vector<double> rs, vs;
		std::string csv = "r,r_over_R,g_radial,g_mag,v,v_kepler,cells\n";
		double vk0 = -1;   // Keplerian reference, normalised to the first valid sample

		for (int n = 0; n < nRadii; n++) {
			double r = rMin + (rMax - rMin) * n / std::max(1, nRadii - 1);
			double sumR = 0, sumM = 0; long cells = 0;
			for (V3 d : dirs) {
				V3 p = c + d * (float)r;
				double lo[3] = { p.x - bx, p.y - by, p.z - bz }, hi[3] = { p.x + bx, p.y + by, p.z + bz };
				if (lo[0] < -Hx || lo[1] < -Hy || lo[2] < -Hz || hi[0] > Hx || hi[1] > Hy || hi[2] > Hz) continue;
				// grid cells whose centres lie inside the AABB
				int i0 = (int)std::ceil((lo[0] + Hx) / cx - 0.5), i1 = (int)std::floor((hi[0] + Hx) / cx - 0.5);
				int j0 = (int)std::ceil((lo[1] + Hy) / cy - 0.5), j1 = (int)std::floor((hi[1] + Hy) / cy - 0.5);
				int k0 = (int)std::ceil((lo[2] + Hz) / cz - 0.5), k1 = (int)std::floor((hi[2] + Hz) / cz - 0.5);
				i0 = std::max(i0, 0); j0 = std::max(j0, 0); k0 = std::max(k0, 0);
				i1 = std::min(i1, Nx - 1); j1 = std::min(j1, Ny - 1); k1 = std::min(k1, Nz - 1);
				for (int k = k0; k <= k1; k++)
					for (int j = j0; j <= j1; j++)
						for (int i = i0; i <= i1; i++) {
							V3 q = { (float)(-Hx + (i + 0.5) * cx), (float)(-Hy + (j + 0.5) * cy), (float)(-Hz + (k + 0.5) * cz) };
							V3 toC = c - q;
							float lq = len(toC);
							if (lq < 1e-6f) continue;
							size_t id = ID(i, j, k);
							V3 g = { gx[id], gy[id], gz[id] };
							sumR += dot(g, toC) / lq;       // inward radial component
							sumM += len(g);
							cells++;
						}
			}
			if (!cells) continue;
			double gr = sumR / cells, gm = sumM / cells;
			double v = std::sqrt(std::max(0.0, gr * r));
			if (vk0 < 0 && v > 0) vk0 = v * std::sqrt(r);
			double vk = vk0 > 0 ? vk0 / std::sqrt(r) : 0;
			printf("  %12.4f %8.3f %14.6e %14.6e %12.6e %12.6e %6ld\n", r, r / R, gr, gm, v, vk, cells);
			char line[256];
			snprintf(line, sizeof(line), "%.6f,%.6f,%.8e,%.8e,%.8e,%.8e,%ld\n", r, r / R, gr, gm, v, vk, cells);
			csv += line;
			if (v > 0) { rs.push_back(std::log(r)); vs.push_back(std::log(v)); }
		}

		// flatness: least-squares slope of log v against log r (0 = flat, -0.5 = Keplerian)
		if (rs.size() >= 2) {
			double mr = 0, mv = 0;
			for (size_t i = 0; i < rs.size(); i++) { mr += rs[i]; mv += vs[i]; }
			mr /= rs.size(); mv /= rs.size();
			double sxy = 0, sxx = 0;
			for (size_t i = 0; i < rs.size(); i++) { sxy += (rs[i] - mr) * (vs[i] - mv); sxx += (rs[i] - mr) * (rs[i] - mr); }
			double slope = sxx > 0 ? sxy / sxx : 0;
			printf("  log-log slope d ln v / d ln r = %+.4f   (0 = flat, -0.5 = Keplerian)\n", slope);
		}
		else printf("  not enough positive samples for a slope\n");

		char name[128];
		snprintf(name, sizeof(name), "rotation_curve_step%llu.csv", (unsigned long long)stepNo);
		if (FILE* f = std::fopen(name, "w")) { std::fputs(csv.c_str(), f); std::fclose(f); printf("  wrote %s\n\n", name); }
		else fprintf(stderr, "  cannot write %s\n", name);
	}
};

// ---------------------------------------------------------------- globals

static Sim* gSim = nullptr;
static std::thread gSimThread;
static std::atomic<bool> gRunning(true);
static std::atomic<bool> gPaused(false);
static std::atomic<bool> gRotReq(false);   // set by key r, served by the sim thread between steps
static std::atomic<bool> gMassReq(false);  // set by key m, served by the sim thread between steps

static void simLoop()
{
	while (gRunning) {
		if (gRotReq.exchange(false)) gSim->computeRotationCurve();
		if (gMassReq.exchange(false)) gSim->computeMass();
		if (gPaused) { std::this_thread::sleep_for(std::chrono::milliseconds(20)); continue; }
		gSim->step();
	}
}

// ---------------------------------------------------------------- rendering

struct M4 { float m[16]; };
static M4 mul(const M4& a, const M4& b)
{
	M4 r;
	for (int c = 0; c < 4; c++)
		for (int rr = 0; rr < 4; rr++) {
			float s = 0;
			for (int k = 0; k < 4; k++) s += a.m[k * 4 + rr] * b.m[c * 4 + k];
			r.m[c * 4 + rr] = s;
		}
	return r;
}
static M4 perspective(float fovy, float aspect, float n, float f)
{
	M4 r{}; float t = 1.0f / std::tan(fovy * 0.5f);
	r.m[0] = t / aspect; r.m[5] = t; r.m[10] = (f + n) / (n - f); r.m[11] = -1; r.m[14] = 2 * f * n / (n - f);
	return r;
}
static M4 inverse(const M4& M)
{
	const float* m = M.m; float inv[16];
	inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
	inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
	inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
	inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
	inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
	inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
	inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
	inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
	inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
	inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
	inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
	inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
	inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
	inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
	inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
	inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
	float det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
	M4 r{}; if (det == 0) return r;
	for (int i = 0; i < 16; i++) r.m[i] = inv[i] / det;
	return r;
}
static M4 lookAt(V3 eye, V3 at, V3 up)
{
	V3 f = norm(at - eye), s = norm(cross(f, up)), u = cross(s, f);
	M4 r{};
	r.m[0] = s.x; r.m[4] = s.y; r.m[8] = s.z;
	r.m[1] = u.x; r.m[5] = u.y; r.m[9] = u.z;
	r.m[2] = -f.x; r.m[6] = -f.y; r.m[10] = -f.z;
	r.m[12] = -dot(s, eye); r.m[13] = -dot(u, eye); r.m[14] = dot(f, eye); r.m[15] = 1;
	return r;
}

static const char* kSegVS = R"(#version 430
layout(location=0) in vec3 aPos;
layout(location=1) in float aBend;
layout(location=2) in float aEm;
uniform mat4 uMVP; uniform vec3 uH; uniform sampler3D uDens;
uniform float uBendScale; uniform int uColorMode;
uniform int uPass;            // 0 = faint cloud, 1 = highlighted lines (opaque), 2 = field streamlines
out vec4 vCol;
vec3 ramp(float t){
	t = clamp(t, 0.0, 1.0);
	vec3 c0 = vec3(0.10, 0.15, 0.85), c1 = vec3(0.10, 0.80, 0.95), c2 = vec3(0.20, 0.95, 0.30),
	     c3 = vec3(1.00, 0.90, 0.10), c4 = vec3(1.00, 0.15, 0.05);
	float s = t * 4.0;
	if (s < 1.0) return mix(c0, c1, s);
	if (s < 2.0) return mix(c1, c2, s - 1.0);
	if (s < 3.0) return mix(c2, c3, s - 2.0);
	return mix(c3, c4, s - 3.0);
}
void main(){
	vec3 base;
	if (uColorMode == 1) base = ramp(aBend / uBendScale);
	else base = mix(vec3(1.0, 0.55, 0.2), vec3(0.3, 0.7, 1.0), clamp(aEm, 0.0, 1.0));
	if (uPass == 2) vCol = vec4(ramp(aBend), 0.8);      // aBend already holds 0..1
	else if (uPass == 1) vCol = vec4(base, 1.0);
	else {
		float d = texture(uDens, (aPos + uH) / (2.0 * uH)).r;
		vCol = vec4(base * clamp(0.35 + 0.65 * sqrt(d), 0.0, 1.0), 0.01 /* field line opacity*/);
	}
	gl_Position = uMVP * vec4(aPos, 1.0);
})";
static const char* kSegFS = R"(#version 430
in vec4 vCol; out vec4 oCol;
void main(){ oCol = vCol; })";
static const char* kSimpleVS = R"(#version 430
layout(location=0) in vec3 aPos;
uniform mat4 uMVP; uniform vec3 uC; uniform vec3 uR;
void main(){ gl_Position = uMVP * vec4(uC + aPos * uR, 1.0); })";
static const char* kSimpleFS = R"(#version 430
uniform vec4 uCol; out vec4 oCol;
void main(){ oCol = uCol; })";

// Volume rendering of the density texture: a full-screen triangle; each fragment
// reconstructs its world-space view ray from the inverse MVP, clips it to the grid box
// and ray-marches the 3D texture front to back (emission-absorption).
static const char* kVolVS = R"(#version 430
out vec2 vNdc;
void main(){
	vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2) * 2.0 - 1.0;   // covers the screen
	vNdc = p;
	gl_Position = vec4(p, 0.0, 1.0);
})";
static const char* kVolFS = R"(#version 430
in vec2 vNdc; out vec4 oCol;
uniform mat4 uInvMVP; uniform vec3 uH; uniform sampler3D uDens;
uniform float uGain;        // brightness / opacity multiplier
uniform int uSteps;         // samples along the box diagonal
vec3 ramp(float t){
	t = clamp(t, 0.0, 1.0);
	vec3 c0 = vec3(0.05, 0.02, 0.20), c1 = vec3(0.35, 0.10, 0.60), c2 = vec3(0.90, 0.30, 0.30),
	     c3 = vec3(1.00, 0.75, 0.30), c4 = vec3(1.00, 1.00, 0.90);
	float s = t * 4.0;
	if (s < 1.0) return mix(c0, c1, s);
	if (s < 2.0) return mix(c1, c2, s - 1.0);
	if (s < 3.0) return mix(c2, c3, s - 2.0);
	return mix(c3, c4, s - 3.0);
}
void main(){
	vec4 a = uInvMVP * vec4(vNdc, -1.0, 1.0), b = uInvMVP * vec4(vNdc, 1.0, 1.0);
	vec3 ro = a.xyz / a.w, rd = normalize(b.xyz / b.w - ro);
	// slab intersection with [-H, H]
	vec3 inv = 1.0 / rd;
	vec3 t0 = (-uH - ro) * inv, t1 = (uH - ro) * inv;
	vec3 tn = min(t0, t1), tf = max(t0, t1);
	float tNear = max(max(tn.x, tn.y), max(tn.z, 0.0));
	float tFar = min(min(tf.x, tf.y), tf.z);
	if (tFar <= tNear) discard;
	float dt = 2.0 * length(uH) / float(uSteps);
	float unit = 2.0 * max(uH.x, max(uH.y, uH.z)) / 100.0;   // opacity per 1/100 of the box
	// jitter the start to hide slicing artefacts
	float j = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
	vec3 col = vec3(0.0); float alpha = 0.0;
	for (float t = tNear + j * dt; t < tFar; t += dt) {
		vec3 p = ro + rd * t;
		float d = texture(uDens, (p + uH) / (2.0 * uH)).r;
		float v = sqrt(max(d, 0.0));
		float aS = 1.0 - exp(-v * uGain * dt / unit);
		col += (1.0 - alpha) * aS * ramp(v);
		alpha += (1.0 - alpha) * aS;
		if (alpha > 0.99) break;
	}
	oCol = vec4(col, alpha);   // premultiplied
})";

static GLuint gVolProg = 0, gVolVAO = 0;
static bool gShowVolume = true;      // draw the density volume
static bool gShowCloud = true;       // draw the faint transparent field lines
static float gVolGain = 0.3f;

static GLuint gSegProg, gSimpleProg, gSegVAO, gSegVBO, gSphereVAO, gBoxVAO, gDensTex;
static GLsizei gSegVerts = 0, gHlStart = 0, gSphereVerts = 0, gBoxVerts = 0;
static int gW = 1280, gH = 800;
static float gYaw = 0.6f, gPitch = 0.4f, gDist = 1;
static int gMouseBtn = -1, gMouseX = 0, gMouseY = 0;
static int gColorMode = 1;
static float gBendScale = 1.0f, gBendGain = 1.0f;
static bool gShowHighlight = true;   // draw highlighted lines at full opacity
static GLuint gFieldVAO = 0, gFieldVBO = 0;
static GLsizei gFieldVerts = 0;
static bool gShowField = true;       // draw the exterior field streamlines (sun mode)

static GLuint compile(GLenum type, const char* src)
{
	GLuint s = glCreateShader(type);
	glShaderSource(s, 1, &src, nullptr);
	glCompileShader(s);
	GLint ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
	if (!ok) { char log[4096]; glGetShaderInfoLog(s, 4096, nullptr, log); fprintf(stderr, "Shader error:\n%s\n", log); std::exit(1); }
	return s;
}
static GLuint link(std::vector<GLuint> shaders)
{
	GLuint p = glCreateProgram();
	for (GLuint s : shaders) glAttachShader(p, s);
	glLinkProgram(p);
	GLint ok; glGetProgramiv(p, GL_LINK_STATUS, &ok);
	if (!ok) { char log[4096]; glGetProgramInfoLog(p, 4096, nullptr, log); fprintf(stderr, "Link error:\n%s\n", log); std::exit(1); }
	for (GLuint s : shaders) glDeleteShader(s);
	return p;
}

static GLuint makeLineVAO(const std::vector<float>& v, GLsizei& count)
{
	GLuint vao, vbo;
	glGenVertexArrays(1, &vao); glBindVertexArray(vao);
	glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, v.size() * sizeof(float), v.data(), GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, (void*)0);
	count = (GLsizei)(v.size() / 3);
	return vao;
}

// ---------------------------------------------------------------- GPU grid solver (compute shaders)

static const char* kCSHeader = R"(#version 430
layout(local_size_x = 8, local_size_y = 8, local_size_z = 4) in;
uniform ivec3 uN;
uint ID(int i, int j, int k) { return uint(i) + uint(uN.x) * (uint(j) + uint(uN.y) * uint(k)); }
)";

// Dirichlet boundary: monopole potential of the total mass.
static const char* kBoundCS = R"(
layout(std430, binding = 0) buffer Phi { float phi[]; };
uniform vec3 uH; uniform vec3 uC; uniform float uM;
void main() {
	ivec3 p = ivec3(gl_GlobalInvocationID);
	if (any(greaterThanEqual(p, uN))) return;
	if (all(greaterThan(p, ivec3(0))) && all(lessThan(p, uN - 1))) return;
	vec3 x = -uH + (vec3(p) + 0.5) * uC;
	phi[ID(p.x, p.y, p.z)] = -uM / length(x);
})";

// One colour of red-black SOR for lap(phi) = 4 pi rho. x is compacted: each invocation
// handles one cell of the active colour.
static const char* kSorCS = R"(
layout(std430, binding = 0) buffer Phi { float phi[]; };
layout(std430, binding = 1) readonly buffer Rho { float rho[]; };
uniform vec3 uA; uniform float uInvDiag; uniform float uW; uniform int uColor;
void main() {
	int j = int(gl_GlobalInvocationID.y) + 1, k = int(gl_GlobalInvocationID.z) + 1;
	int i = 1 + ((j + k + uColor + 1) & 1) + 2 * int(gl_GlobalInvocationID.x);
	if (i >= uN.x - 1 || j >= uN.y - 1 || k >= uN.z - 1) return;
	uint c = ID(i, j, k), sy = uint(uN.x), sz = uint(uN.x) * uint(uN.y);
	float nb = uA.x * (phi[c - 1u] + phi[c + 1u])
	         + uA.y * (phi[c - sy] + phi[c + sy])
	         + uA.z * (phi[c - sz] + phi[c + sz]);
	float gs = (nb - 12.566370614359172 * rho[c]) * uInvDiag;
	phi[c] += uW * (gs - phi[c]);
})";

// Gradient g = -grad(phi) and max |g|^2 (workgroup shared atomic, then one global atomic).
// Non-negative floats order like their bit patterns, so atomicMax on uint works.
static const char* kGradCS = R"(
layout(std430, binding = 0) readonly buffer Phi { float phi[]; };
layout(std430, binding = 2) writeonly buffer GX { float gx[]; };
layout(std430, binding = 3) writeonly buffer GY { float gy[]; };
layout(std430, binding = 4) writeonly buffer GZ { float gz[]; };
layout(std430, binding = 5) buffer GMax { uint gmax; };
uniform vec3 uInvC;
shared uint smax;
void main() {
	if (gl_LocalInvocationIndex == 0u) smax = 0u;
	barrier();
	ivec3 p = ivec3(gl_GlobalInvocationID);
	if (all(lessThan(p, uN))) {
		ivec3 hi = min(p + 1, uN - 1), lo = max(p - 1, ivec3(0));
		float x = -(phi[ID(hi.x, p.y, p.z)] - phi[ID(lo.x, p.y, p.z)]) * uInvC.x / float(hi.x - lo.x);
		float y = -(phi[ID(p.x, hi.y, p.z)] - phi[ID(p.x, lo.y, p.z)]) * uInvC.y / float(hi.y - lo.y);
		float z = -(phi[ID(p.x, p.y, hi.z)] - phi[ID(p.x, p.y, lo.z)]) * uInvC.z / float(hi.z - lo.z);
		uint c = ID(p.x, p.y, p.z);
		gx[c] = x; gy[c] = y; gz[c] = z;
		atomicMax(smax, floatBitsToUint(x * x + y * y + z * z));
	}
	barrier();
	if (gl_LocalInvocationIndex == 0u) atomicMax(gmax, smax);
})";

// Normalised density straight into the 3D texture used by the line shader.
static const char* kDensCS = R"(
layout(std430, binding = 1) readonly buffer Rho { float rho[]; };
layout(r32f, binding = 0) writeonly uniform image3D uImg;
uniform float uInv;
void main() {
	ivec3 p = ivec3(gl_GlobalInvocationID);
	if (any(greaterThanEqual(p, uN))) return;
	imageStore(uImg, p, vec4(rho[ID(p.x, p.y, p.z)] * uInv));
})";

struct GpuSolver {
	bool ok = false;
	GLuint progBound = 0, progSor = 0, progGrad = 0, progDens = 0;
	GLuint bPhi = 0, bRho = 0, bG[3] = { 0, 0, 0 }, bMax = 0;
};
static GpuSolver gGpu;

static GLuint makeCompute(const char* body)
{
	std::string src = std::string(kCSHeader) + body;
	return link({ compile(GL_COMPUTE_SHADER, src.c_str()) });
}
static inline GLuint groups(int n, int g) { return (GLuint)((std::max(n, 0) + g - 1) / g); }

static bool gpuInit(Sim& s)
{
	size_t bytes = s.rho.size() * sizeof(float);
	GLint64 maxBlock = 0;
	glGetInteger64v(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &maxBlock);
	if ((GLint64)bytes > maxBlock) {
		fprintf(stderr, "GPU: grid needs %.0f MB per buffer, driver limit %.0f MB -> CPU solver\n",
			bytes / 1048576.0, maxBlock / 1048576.0);
		return false;
	}
	gGpu.progBound = makeCompute(kBoundCS);
	gGpu.progSor = makeCompute(kSorCS);
	gGpu.progGrad = makeCompute(kGradCS);
	gGpu.progDens = makeCompute(kDensCS);

	auto mk = [&](GLuint& b, size_t n) {
		glGenBuffers(1, &b);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, b);
		glBufferData(GL_SHADER_STORAGE_BUFFER, n, nullptr, GL_DYNAMIC_COPY);
		float zero = 0;
		glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, &zero);
		};
	mk(gGpu.bPhi, bytes); mk(gGpu.bRho, bytes);
	for (auto& b : gGpu.bG) mk(b, bytes);
	mk(gGpu.bMax, sizeof(GLuint));
	if (glGetError() != GL_NO_ERROR) { fprintf(stderr, "GPU: buffer allocation failed -> CPU solver\n"); return false; }

	printf("GPU grid solver: %s (%s), %.2f GB of GPU buffers\n",
		(const char*)glGetString(GL_RENDERER), (const char*)glGetString(GL_VERSION), 5.0 * bytes / 1073741824.0);
	gGpu.ok = true;
	return true;
}

// Must run on the thread that owns the GL context.
static void gpuSolve(Sim& s)
{
	const int Nx = s.Nx, Ny = s.Ny, Nz = s.Nz;
	size_t bytes = s.rho.size() * sizeof(float);
	GLuint zeroU = 0;

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, gGpu.bRho);
	glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, bytes, s.rho.data());
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, gGpu.bMax);
	glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &zeroU);

	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, gGpu.bPhi);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, gGpu.bRho);
	for (int a = 0; a < 3; a++) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2 + a, gGpu.bG[a]);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, gGpu.bMax);

	auto setN = [&](GLuint p) { glUseProgram(p); glUniform3i(glGetUniformLocation(p, "uN"), Nx, Ny, Nz); };

	// boundary
	setN(gGpu.progBound);
	glUniform3f(glGetUniformLocation(gGpu.progBound, "uH"), (float)s.Hx, (float)s.Hy, (float)s.Hz);
	glUniform3f(glGetUniformLocation(gGpu.progBound, "uC"), (float)s.cx, (float)s.cy, (float)s.cz);
	glUniform1f(glGetUniformLocation(gGpu.progBound, "uM"), (float)s.massTotal);
	glDispatchCompute(groups(Nx, 8), groups(Ny, 8), groups(Nz, 4));
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

	// red-black SOR
	const float ax = (float)(1.0 / (s.cx * s.cx)), ay = (float)(1.0 / (s.cy * s.cy)), az = (float)(1.0 / (s.cz * s.cz));
	setN(gGpu.progSor);
	glUniform3f(glGetUniformLocation(gGpu.progSor, "uA"), ax, ay, az);
	glUniform1f(glGetUniformLocation(gGpu.progSor, "uInvDiag"), 1.0f / (2.0f * (ax + ay + az)));
	glUniform1f(glGetUniformLocation(gGpu.progSor, "uW"), 1.85f);
	GLint uColor = glGetUniformLocation(gGpu.progSor, "uColor");
	GLuint sgx = groups((Nx - 2 + 1) / 2, 8), sgy = groups(Ny - 2, 8), sgz = groups(Nz - 2, 4);
	for (int it = 0; it < s.P.iters; it++)
		for (int color = 0; color < 2; color++) {
			glUniform1i(uColor, color);
			glDispatchCompute(sgx, sgy, sgz);
			glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
		}

	// gradient + max reduction
	setN(gGpu.progGrad);
	glUniform3f(glGetUniformLocation(gGpu.progGrad, "uInvC"), (float)(1.0 / s.cx), (float)(1.0 / s.cy), (float)(1.0 / s.cz));
	glDispatchCompute(groups(Nx, 8), groups(Ny, 8), groups(Nz, 4));

	// density texture for the renderer
	if (gDensTex) {
		setN(gGpu.progDens);
		glUniform1f(glGetUniformLocation(gGpu.progDens, "uInv"), s.rhoRef > 0 ? 1.0f / s.rhoRef : 0.0f);
		glBindImageTexture(0, gDensTex, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F);
		glDispatchCompute(groups(Nx, 8), groups(Ny, 8), groups(Nz, 4));
	}
	glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

	// read back what the CPU line pass needs
	std::vector<float>* dst[3] = { &s.gx, &s.gy, &s.gz };
	for (int a = 0; a < 3; a++) {
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, gGpu.bG[a]);
		glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, bytes, dst[a]->data());
	}
	if (s.P.mode == SUN) {   // the Shapiro-delay measurement samples the potential on the CPU
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, gGpu.bPhi);
		glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, bytes, s.phi.data());
	}
	GLuint m2 = 0;
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, gGpu.bMax);
	glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint), &m2);
	float g2; std::memcpy(&g2, &m2, 4);
	s.gRef = std::sqrt(g2);
	glUseProgram(0);
}

// Sim thread -> GL thread hand-off.
static std::mutex gGpuMtx;
static std::condition_variable gGpuCv;
static bool gGpuReq = false;

static void gpuRequestBlocking()
{
	std::unique_lock<std::mutex> l(gGpuMtx);
	gGpuReq = true;
	gGpuCv.wait(l, [] { return !gGpuReq || !gRunning; });
}
static bool serviceGpu()
{
	{ std::lock_guard<std::mutex> l(gGpuMtx); if (!gGpuReq) return false; }
	gpuSolve(*gSim);
	{ std::lock_guard<std::mutex> l(gGpuMtx); gGpuReq = false; }
	gGpuCv.notify_all();
	return true;
}

static void initGL()
{
	gSegProg = link({ compile(GL_VERTEX_SHADER, kSegVS), compile(GL_FRAGMENT_SHADER, kSegFS) });
	gSimpleProg = link({ compile(GL_VERTEX_SHADER, kSimpleVS), compile(GL_FRAGMENT_SHADER, kSimpleFS) });
	gVolProg = link({ compile(GL_VERTEX_SHADER, kVolVS), compile(GL_FRAGMENT_SHADER, kVolFS) });
	glGenVertexArrays(1, &gVolVAO);   // core profile needs a VAO bound even with no attributes

	glGenVertexArrays(1, &gSegVAO); glBindVertexArray(gSegVAO);
	glGenBuffers(1, &gSegVBO); glBindBuffer(GL_ARRAY_BUFFER, gSegVBO);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(DispVert), (void*)0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(DispVert), (void*)12);
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(DispVert), (void*)16);

	// same vertex layout for the field streamlines
	glGenVertexArrays(1, &gFieldVAO); glBindVertexArray(gFieldVAO);
	glGenBuffers(1, &gFieldVBO); glBindBuffer(GL_ARRAY_BUFFER, gFieldVBO);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(DispVert), (void*)0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(DispVert), (void*)12);
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(DispVert), (void*)16);

	// Fewer wireframe lines per sphere when there are many emitters.
	bool many = gSim->nE > 8;
	const int LAT = many ? 6 : 12, LON = many ? 8 : 24, SEGS = many ? 16 : 48;
	std::vector<float> s;
	for (int i = 1; i < LAT; i++) {
		float th = (float)(PI * i / LAT);
		for (int j = 0; j < SEGS; j++) {
			float a0 = (float)(2 * PI * j / SEGS), a1 = (float)(2 * PI * (j + 1) / SEGS);
			s.insert(s.end(), { std::sin(th) * std::cos(a0), std::cos(th), std::sin(th) * std::sin(a0),
				std::sin(th) * std::cos(a1), std::cos(th), std::sin(th) * std::sin(a1) });
		}
	}
	for (int j = 0; j < LON; j++) {
		float a = (float)(2 * PI * j / LON);
		for (int i = 0; i < SEGS; i++) {
			float t0 = (float)(PI * i / SEGS), t1 = (float)(PI * (i + 1) / SEGS);
			s.insert(s.end(), { std::sin(t0) * std::cos(a), std::cos(t0), std::sin(t0) * std::sin(a),
				std::sin(t1) * std::cos(a), std::cos(t1), std::sin(t1) * std::sin(a) });
		}
	}
	gSphereVAO = makeLineVAO(s, gSphereVerts);

	std::vector<float> b;
	for (int i = 0; i < 4; i++) {
		float x = (i & 1) ? 1.f : -1.f, y = (i & 2) ? 1.f : -1.f;
		b.insert(b.end(), { x, y, -1, x, y, 1,  x, -1, y, x, 1, y,  -1, x, y, 1, x, y });
	}
	gBoxVAO = makeLineVAO(b, gBoxVerts);

	glGenTextures(1, &gDensTex);
	glBindTexture(GL_TEXTURE_3D, gDensTex);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	std::vector<float> zero((size_t)gSim->Nx * gSim->Ny * gSim->Nz, 0.0f);
	glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, gSim->Nx, gSim->Ny, gSim->Nz, 0, GL_RED, GL_FLOAT, zero.data());

	gDist = (float)(3.0 * gSim->Hmax);
}

static void renderScene(int w, int h);

static void display()
{
	{
		std::lock_guard<std::mutex> lock(gSim->dispMtx);
		if (gSim->dispNew) {
			glBindBuffer(GL_ARRAY_BUFFER, gSegVBO);
			glBufferData(GL_ARRAY_BUFFER, gSim->dispVerts.size() * sizeof(DispVert), gSim->dispVerts.data(), GL_STREAM_DRAW);
			gSegVerts = (GLsizei)gSim->dispVerts.size();
			gHlStart = (GLsizei)gSim->dispHlStart;
			glBindBuffer(GL_ARRAY_BUFFER, gFieldVBO);
			glBufferData(GL_ARRAY_BUFFER, gSim->dispField.size() * sizeof(DispVert), gSim->dispField.data(), GL_STREAM_DRAW);
			gFieldVerts = (GLsizei)gSim->dispField.size();
			if (!gSim->dispRho.empty()) {
				glBindTexture(GL_TEXTURE_3D, gDensTex);
				glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, gSim->Nx, gSim->Ny, gSim->Nz, GL_RED, GL_FLOAT, gSim->dispRho.data());
			}
			gBendScale = gSim->dispBendScale;
			gSim->dispNew = false;
			char title[256];
			const char* mode = gSim->P.mode == GALAXY ? "galaxy" : gSim->P.mode == SINGLE ? "1 emitter" : gSim->P.mode == SUN ? "sun" : "2 emitters";
			if (gColorMode)
				snprintf(title, sizeof(title), "Field lines (%s) | step %llu | %.3e nodes (%s) | %.2fs/step | colour = bend (red = %.3g rad)%s",
					mode, (unsigned long long)gSim->stepNo, (double)gSim->count, gSim->disk ? "disk" : "RAM",
					gSim->lastStepSeconds, gBendScale / gBendGain, gPaused ? " | PAUSED" : "");
			else
				snprintf(title, sizeof(title), "Field lines (%s) | step %llu | %.3e nodes (%s) | %.2fs/step | colour = emitter%s",
					mode, (unsigned long long)gSim->stepNo, (double)gSim->count, gSim->disk ? "disk" : "RAM",
					gSim->lastStepSeconds, gPaused ? " | PAUSED" : "");
			glutSetWindowTitle(title);
		}
	}

	renderScene(gW, gH);
	glutSwapBuffers();
}

// Draw the whole scene into the currently bound framebuffer at w x h pixels.
static void renderScene(int w, int h)
{
	glViewport(0, 0, w, h);
	glClearColor(1.0f, 1.0f, 1.0f, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	float Hx = (float)gSim->Hx, Hy = (float)gSim->Hy, Hz = (float)gSim->Hz, Hm = (float)gSim->Hmax;
	V3 eye = { gDist * std::cos(gPitch) * std::sin(gYaw), gDist * std::sin(gPitch), gDist * std::cos(gPitch) * std::cos(gYaw) };
	M4 mvp = mul(perspective(0.8f, (float)w / h, Hm * 0.01f, Hm * 40.0f), lookAt(eye, { 0, 0, 0 }, { 0, 1, 0 }));

	glEnable(GL_BLEND);
	glDisable(GL_DEPTH_TEST);

	// density volume first (premultiplied colour), lines and wireframes add on top
	if (gShowVolume) {
		glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
		glUseProgram(gVolProg);
		M4 inv = inverse(mvp);
		glUniformMatrix4fv(glGetUniformLocation(gVolProg, "uInvMVP"), 1, GL_FALSE, inv.m);
		glUniform3f(glGetUniformLocation(gVolProg, "uH"), Hx, Hy, Hz);
		glUniform1f(glGetUniformLocation(gVolProg, "uGain"), gVolGain);
		int nMax = std::max(gSim->Nx, std::max(gSim->Ny, gSim->Nz));
		glUniform1i(glGetUniformLocation(gVolProg, "uSteps"), std::min(1024, std::max(64, 2 * nMax)));
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_3D, gDensTex);
		glUniform1i(glGetUniformLocation(gVolProg, "uDens"), 0);
		glBindVertexArray(gVolVAO);
		glDrawArrays(GL_TRIANGLES, 0, 3);
	}
	glBlendFunc(GL_SRC_ALPHA, GL_ONE);

	glUseProgram(gSimpleProg);
	glUniformMatrix4fv(glGetUniformLocation(gSimpleProg, "uMVP"), 1, GL_FALSE, mvp.m);
	GLint uC = glGetUniformLocation(gSimpleProg, "uC"), uR = glGetUniformLocation(gSimpleProg, "uR"), uCol = glGetUniformLocation(gSimpleProg, "uCol");

	glBindVertexArray(gBoxVAO);
	glUniform3f(uC, 0, 0, 0); glUniform3f(uR, Hx, Hy, Hz); glUniform4f(uCol, 0.4f, 0.4f, 0.5f, 0.5f);
	glDrawArrays(GL_LINES, 0, gBoxVerts);

	glBindVertexArray(gSphereVAO);
	float r = (float)gSim->R;
	uint32_t nE = gSim->nE;
	float sphereAlpha = nE > 8 ? 0.2f : 0.35f;
	for (uint32_t e = 0; e < nE; e++) {
		V3 c = gSim->centers[e];
		float t = nE > 1 ? (float)e / (float)(nE - 1) : 0.0f;   // orange -> blue across emitters
		glUniform3f(uC, c.x, c.y, c.z); glUniform3f(uR, r, r, r);
		glUniform4f(uCol, 1.0f + t * (0.4f - 1.0f), 0.7f + t * (0.8f - 0.7f), 0.3f + t * (1.0f - 0.3f), sphereAlpha);
		glDrawArrays(GL_LINES, 0, gSphereVerts);
	}

	glUseProgram(gSegProg);
	glUniformMatrix4fv(glGetUniformLocation(gSegProg, "uMVP"), 1, GL_FALSE, mvp.m);
	glUniform3f(glGetUniformLocation(gSegProg, "uH"), Hx, Hy, Hz);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_3D, gDensTex);
	glUniform1i(glGetUniformLocation(gSegProg, "uDens"), 0);
	glUniform1f(glGetUniformLocation(gSegProg, "uBendScale"), gBendScale / gBendGain);
	glUniform1i(glGetUniformLocation(gSegProg, "uColorMode"), gColorMode);
	GLint uPass = glGetUniformLocation(gSegProg, "uPass");
	glBindVertexArray(gSegVAO);
	if (gSegVerts) {
		// faint cloud (with h off, the highlighted lines join the cloud as before)
		GLsizei nCloud = gShowHighlight ? gHlStart : gSegVerts;
		if (gShowCloud && nCloud > 0) {
			glUniform1i(uPass, 0);
			glDrawArrays(GL_LINES, 0, nCloud);
		}
		if (gShowHighlight && gSegVerts > gHlStart) {
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			glUniform1i(uPass, 1);
			glDrawArrays(GL_LINES, gHlStart, gSegVerts - gHlStart);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE);
		}
	}
	if (gShowField && gFieldVerts) {   // exterior field streamlines (sun mode)
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glUniform1i(uPass, 2);
		glBindVertexArray(gFieldVAO);
		glDrawArrays(GL_LINES, 0, gFieldVerts);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE);
	}
}

// ---------------------------------------------------------------- PNG screenshot
// Minimal dependency-free PNG writer (zlib "stored" deflate blocks, no compression library).

static uint32_t crc32Png(const unsigned char* d, size_t n, uint32_t c = 0xFFFFFFFFu)
{
	static uint32_t table[256];
	static bool init = false;
	if (!init) {
		for (uint32_t i = 0; i < 256; i++) {
			uint32_t v = i;
			for (int k = 0; k < 8; k++) v = (v & 1) ? 0xEDB88320u ^ (v >> 1) : v >> 1;
			table[i] = v;
		}
		init = true;
	}
	for (size_t i = 0; i < n; i++) c = table[(c ^ d[i]) & 0xFF] ^ (c >> 8);
	return c;
}

static bool writePNG(const char* path, int w, int h, const std::vector<unsigned char>& rgb)
{
	FILE* f = std::fopen(path, "wb");
	if (!f) return false;
	auto be32 = [](unsigned char* p, uint32_t v) { p[0] = v >> 24; p[1] = v >> 16; p[2] = v >> 8; p[3] = (unsigned char)v; };
	auto chunk = [&](const char* type, const unsigned char* data, size_t n) {
		unsigned char hdr[8]; be32(hdr, (uint32_t)n); std::memcpy(hdr + 4, type, 4);
		std::fwrite(hdr, 1, 8, f);
		if (n) std::fwrite(data, 1, n, f);
		uint32_t c = crc32Png(hdr + 4, 4);
		c = crc32Png(data, n, c) ^ 0xFFFFFFFFu;
		unsigned char cb[4]; be32(cb, c); std::fwrite(cb, 1, 4, f);
		};

	static const unsigned char sig[8] = { 137, 80, 78, 71, 13, 10, 26, 10 };
	std::fwrite(sig, 1, 8, f);

	unsigned char ihdr[13];
	be32(ihdr, (uint32_t)w); be32(ihdr + 4, (uint32_t)h);
	ihdr[8] = 8; ihdr[9] = 2; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;   // 8-bit RGB
	chunk("IHDR", ihdr, 13);

	// raw scanlines: filter byte 0 + RGB
	size_t rowBytes = (size_t)w * 3;
	std::vector<unsigned char> raw((rowBytes + 1) * h);
	for (int y = 0; y < h; y++) {
		raw[y * (rowBytes + 1)] = 0;
		std::memcpy(&raw[y * (rowBytes + 1) + 1], &rgb[(size_t)y * rowBytes], rowBytes);
	}

	// zlib stream with stored blocks
	std::vector<unsigned char> z;
	z.reserve(raw.size() + raw.size() / 65535 * 5 + 16);
	z.push_back(0x78); z.push_back(0x01);
	size_t pos = 0;
	do {
		size_t n = std::min<size_t>(65535, raw.size() - pos);
		bool last = pos + n == raw.size();
		z.push_back(last ? 1 : 0);
		z.push_back(n & 0xFF); z.push_back((n >> 8) & 0xFF);
		z.push_back(~n & 0xFF); z.push_back((~n >> 8) & 0xFF);
		z.insert(z.end(), raw.begin() + pos, raw.begin() + pos + n);
		pos += n;
	} while (pos < raw.size());
	uint32_t a = 1, b = 0;
	for (unsigned char c : raw) { a = (a + c) % 65521; b = (b + a) % 65521; }
	unsigned char ad[4]; be32(ad, (b << 16) | a);
	z.insert(z.end(), ad, ad + 4);

	chunk("IDAT", z.data(), z.size());
	chunk("IEND", nullptr, 0);
	bool ok = !std::ferror(f);
	std::fclose(f);
	return ok;
}

static int gShotScale = 4;   // screenshot size = window size * this (clamped to GPU limits)

// Render the current view offscreen at gShotScale x the window size and save it as a PNG.
// Must run on the GL thread (called from the keyboard callback).
static void saveScreenshot()
{
	GLint maxRB = 0, maxVP[2] = { 0, 0 };
	glGetIntegerv(GL_MAX_RENDERBUFFER_SIZE, &maxRB);
	glGetIntegerv(GL_MAX_VIEWPORT_DIMS, maxVP);
	int lim = std::min(maxRB, std::min(maxVP[0], maxVP[1]));
	double s = gShotScale;
	s = std::min(s, (double)lim / gW);
	s = std::min(s, (double)lim / gH);
	int w = std::max(1, (int)(gW * s)), h = std::max(1, (int)(gH * s));

	GLuint fbo, color, depth;
	glGenFramebuffers(1, &fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	glGenRenderbuffers(1, &color);
	glBindRenderbuffer(GL_RENDERBUFFER, color);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, w, h);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, color);
	glGenRenderbuffers(1, &depth);
	glBindRenderbuffer(GL_RENDERBUFFER, depth);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth);

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		fprintf(stderr, "Screenshot: framebuffer %d x %d incomplete\n", w, h);
	}
	else {
		renderScene(w, h);
		glFinish();
		std::vector<unsigned char> px((size_t)w * h * 3), flipped(px.size());
		glPixelStorei(GL_PACK_ALIGNMENT, 1);
		glReadBuffer(GL_COLOR_ATTACHMENT0);
		glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, px.data());
		size_t row = (size_t)w * 3;
		for (int y = 0; y < h; y++)   // OpenGL is bottom-up, PNG is top-down
			std::memcpy(&flipped[(size_t)y * row], &px[(size_t)(h - 1 - y) * row], row);

		char name[128];
		std::time_t t = std::time(nullptr);
		std::strftime(name, sizeof(name), "screenshot_%Y%m%d_%H%M%S.png", std::localtime(&t));
		if (writePNG(name, w, h, flipped)) printf("Saved %s (%d x %d)\n", name, w, h);
		else fprintf(stderr, "Screenshot: cannot write %s\n", name);
	}

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glDeleteRenderbuffers(1, &color);
	glDeleteRenderbuffers(1, &depth);
	glDeleteFramebuffers(1, &fbo);
	glViewport(0, 0, gW, gH);
	glutPostRedisplay();
}

static void reshape(int w, int h) { gW = std::max(w, 1); gH = std::max(h, 1); }
static void idle()
{
	// Serve GPU solve requests promptly; redraw at ~60 Hz.
	static auto last = std::chrono::steady_clock::now();
	if (!serviceGpu()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
	auto now = std::chrono::steady_clock::now();
	if (now - last >= std::chrono::milliseconds(16)) { last = now; glutPostRedisplay(); }
}
static void keyboard(unsigned char key, int, int)
{
	if (key == 27) glutLeaveMainLoop();
	if (key == ' ') gPaused = !gPaused;
	if (key == 'c' || key == 'C') gColorMode ^= 1;
	if (key == ']') gBendGain *= 1.25f;
	if (key == '[') gBendGain /= 1.25f;
	if (key == 'h' || key == 'H') gShowHighlight = !gShowHighlight;
	if (key == 't' || key == 'T') { gShowCloud = !gShowCloud; printf("Transparent field lines %s\n", gShowCloud ? "on" : "off"); }
	if (key == 'v' || key == 'V') { gShowVolume = !gShowVolume; printf("Density volume %s\n", gShowVolume ? "on" : "off"); }
	if (key == '.' || key == '>') { gVolGain *= 1.25f; printf("Volume gain %.3g\n", gVolGain); }
	if (key == ',' || key == '<') { gVolGain /= 1.25f; printf("Volume gain %.3g\n", gVolGain); }
	if (key == 'f' || key == 'F') { gShowField = !gShowField; printf("Field streamlines %s\n", gShowField ? "on" : "off"); }
	if (key == 'p' || key == 'P') saveScreenshot();
	if (key == 'r' || key == 'R') { gRotReq = true; printf("Rotation curve requested (computed after the current step)\n"); }
	if (key == 'm' || key == 'M') { gMassReq = true; printf("Mass measure requested (computed after the current step)\n"); }
	if (key == '+' || key == '=') { gShotScale = std::min(16, gShotScale + 1); printf("Screenshot scale %dx\n", gShotScale); }
	if (key == '-' || key == '_') { gShotScale = std::max(1, gShotScale - 1); printf("Screenshot scale %dx\n", gShotScale); }
}
static void mouse(int button, int state, int x, int y)
{
	gMouseBtn = state == GLUT_DOWN ? button : -1;
	gMouseX = x; gMouseY = y;
}
static void motion(int x, int y)
{
	int dx = x - gMouseX, dy = y - gMouseY;
	gMouseX = x; gMouseY = y;
	if (gMouseBtn == GLUT_LEFT_BUTTON) {
		gYaw -= dx * 0.005f;
		gPitch = std::max(-1.5f, std::min(1.5f, gPitch + dy * 0.005f));
	}
	else if (gMouseBtn == GLUT_RIGHT_BUTTON) {
		gDist *= std::exp(dy * 0.005f);
	}
}
static void wheel(int, int dir, int, int) { gDist *= dir > 0 ? 0.9f : 1.1f; }

// ---------------------------------------------------------------- main

int main(int argc, char** argv)
{
	Params P;
	for (int i = 1; i < argc; i++) {
		std::string a = argv[i];
		auto next = [&]() -> std::string { if (i + 1 >= argc) { fprintf(stderr, "Missing value for %s\n", a.c_str()); std::exit(1); } return argv[++i]; };
		if (a == "--n") P.n = std::stod(next());
		else if (a == "--grid") {
			std::string v = next();
			double g[3]; parseTriple(v, g, "--grid");
			for (int d = 0; d < 3; d++) P.N[d] = (int)g[d];
			P.gridAuto = v.find(',') == std::string::npos;
		}
		else if (a == "--margin") parseTriple(next(), P.margin, "--margin");
		else if (a == "--seg") { P.seg = std::stod(next()); P.segSet = true; }
		else if (a == "--strength") { P.strength = std::stod(next()); P.strengthSet = true; }
		else if (a == "--iters") P.iters = std::stoi(next());
		else if (a == "--ram") P.ramGB = std::stod(next());
		else if (a == "--swap") P.swapDir = next();
		else if (a == "--highlight") P.highlight = std::stod(next());
		else if (a == "--cpu") P.cpu = true;
		else if (a == "--galaxy") P.mode = GALAXY;
		else if (a == "--single") P.mode = SINGLE;
		else if (a == "--two") P.mode = TWO;
		else if (a == "--sun") P.mode = SUN;
		else if (a == "--end-at-emitters") P.endAtEmitters = true;
		else if (a == "--emitters") P.emitters = std::stoi(next());
		else if (a == "--galaxy-radius") P.galaxyRadius = std::stod(next());
		else if (a == "--arms") P.arms = std::stoi(next());
		else if (a == "--thickness") P.thickness = std::stod(next());
		else if (a == "--model") {
			std::string m = next();
			P.model = m == "newton" ? NEWTON : m == "classical" ? CLASSICAL : QUANTUM;
		}
		else { fprintf(stderr, "Unknown option %s\n", a.c_str()); return 1; }
	}

	Sim sim(P);
	gSim = &sim;

	glutInit(&argc, argv);
	glutInitContextVersion(4, 3);
	glutInitContextProfile(GLUT_CORE_PROFILE);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	glutInitWindowSize(gW, gH);
	glutCreateWindow("Field lines");
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK) { fprintf(stderr, "GLEW init failed\n"); return 1; }
	glGetError();

	if (P.cpu) printf("Grid solver: CPU (--cpu)\n");
	else if (gpuInit(sim)) sim.runGPU = gpuRequestBlocking;
	else printf("Grid solver: CPU (GPU unavailable)\n");

	initGL();
	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutIdleFunc(idle);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutMouseWheelFunc(wheel);

	gSimThread = std::thread(simLoop);
	glutMainLoop();
	{ std::lock_guard<std::mutex> l(gGpuMtx); gRunning = false; }
	gGpuCv.notify_all();
	gPaused = false;
	if (gSimThread.joinable()) gSimThread.join();
	return 0;
}