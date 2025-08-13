#include "ssapy.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace ssapy {
    void pyExportSSAPy(py::module& m) {
        py::class_<Ellipsoid, std::shared_ptr<Ellipsoid>>(m, "Ellipsoid",
            "An ellipsoid representation for coordinate transformations.\n\n"
            "Represents a reference ellipsoid (such as WGS84) for converting between\n"
            "geodetic coordinates (longitude, latitude, height) and Earth-Centered\n"
            "Earth-Fixed (ECEF) Cartesian coordinates (x, y, z).\n\n"
            "Parameters\n"
            "----------\n"
            "Req : float, optional\n"
            "    Equatorial radius in meters. Default is WGS84 value (6378137.0 m).\n"
            "f : float, optional\n"
            "    Flattening parameter. Default is WGS84 value (1/298.257223563).\n\n"
            "Examples\n"
            "--------\n"
            ">>> ellipsoid = Ellipsoid()  # WGS84 ellipsoid\n"
            ">>> ellipsoid = Ellipsoid(6378137.0, 1/298.257223563)  # Custom ellipsoid"
        )
            .def(py::init<const double, const double>(),
                "Req"_a=6378.137e3, "f"_a=1.0/298.257223563,
                "Initialize ellipsoid with equatorial radius and flattening.\n\n"
                "Parameters\n"
                "----------\n"
                "Req : float\n"
                "    Equatorial radius in meters\n"
                "f : float\n"
                "    Flattening parameter (dimensionless)"
            )
            .def("_sphereToCart",
                [](
                    const Ellipsoid& e,
                    size_t lonarr, size_t latarr, size_t heightarr, size_t size,
                    size_t xarr, size_t yarr, size_t zarr
                ) {
                    double* lonptr = reinterpret_cast<double*>(lonarr);
                    double* latptr = reinterpret_cast<double*>(latarr);
                    double* heightptr = reinterpret_cast<double*>(heightarr);
                    double* xptr = reinterpret_cast<double*>(xarr);
                    double* yptr = reinterpret_cast<double*>(yarr);
                    double* zptr = reinterpret_cast<double*>(zarr);
                    for(int i=0; i<size; i++) {
                        e.sphereToCart(
                            lonptr[i], latptr[i], heightptr[i],
                            xptr[i], yptr[i], zptr[i]
                        );
                    }
                },
                "Internal vectorized spherical to Cartesian coordinate conversion.\n\n"
                "This is a low-level method that operates on memory pointers for\n"
                "efficient batch conversion. Use the Python wrapper sphereToCart()\n"
                "method instead, which provides a more convenient NumPy interface.\n\n"
                "Parameters\n"
                "----------\n"
                "lonarr, latarr, heightarr : memory addresses\n"
                "    Pointers to input longitude, latitude, and height arrays\n"
                "size : int\n"
                "    Number of elements to convert\n"
                "xarr, yarr, zarr : memory addresses\n"
                "    Pointers to output x, y, z coordinate arrays"
            )
            .def("_cartToSphere",
                [](
                    const Ellipsoid& e,
                    size_t xarr, size_t yarr, size_t zarr, size_t size,
                    size_t lonarr, size_t latarr, size_t heightarr
                ) {
                    double* xptr = reinterpret_cast<double*>(xarr);
                    double* yptr = reinterpret_cast<double*>(yarr);
                    double* zptr = reinterpret_cast<double*>(zarr);
                    double* lonptr = reinterpret_cast<double*>(lonarr);
                    double* latptr = reinterpret_cast<double*>(latarr);
                    double* heightptr = reinterpret_cast<double*>(heightarr);
                    for(int i=0; i<size; i++) {
                        e.cartToSphere(
                            xptr[i], yptr[i], zptr[i],
                            lonptr[i], latptr[i], heightptr[i]
                        );
                    }
                },
                "Internal vectorized Cartesian to spherical coordinate conversion.\n\n"
                "This is a low-level method that operates on memory pointers for\n"
                "efficient batch conversion. Use the Python wrapper cartToSphere()\n"
                "method instead, which provides a more convenient NumPy interface.\n\n"
                "Parameters\n"
                "----------\n"
                "xarr, yarr, zarr : memory addresses\n"
                "    Pointers to input x, y, z coordinate arrays\n"
                "size : int\n"
                "    Number of elements to convert\n"
                "lonarr, latarr, heightarr : memory addresses\n"
                "    Pointers to output longitude, latitude, height arrays"
            );

        py::class_<HarrisPriester>(m, "HarrisPriester",
            "Harris-Priester atmospheric density model.\n\n"
            "Implements the Harris-Priester semi-empirical atmospheric density\n"
            "model for calculating atmospheric drag effects on satellites.\n"
            "This model accounts for diurnal density variations and solar activity.\n\n"
            "Parameters\n"
            "----------\n"
            "ellip : Ellipsoid\n"
            "    Reference ellipsoid for coordinate transformations\n"
            "n : float, optional\n"
            "    Density variation parameter (default: 3.0)\n\n"
            "References\n"
            "----------\n"
            "Harris, I. and Priester, W. (1962). Time-dependent structure of the\n"
            "upper atmosphere. Journal of Atmospheric Sciences, 19(4), 286-301."
        )
            .def(py::init<const Ellipsoid&, const double>(),
                "ellip"_a, "n"_a=3.0, py::keep_alive<1, 2>(),
                "Initialize Harris-Priester atmospheric model.\n\n"
                "Parameters\n"
                "----------\n"
                "ellip : Ellipsoid\n"
                "    Reference ellipsoid for coordinate transformations\n"
                "n : float\n"
                "    Density variation parameter (typically 2-6)"
            )
            .def("density", &HarrisPriester::density,
                "Calculate atmospheric density at given position and time.\n\n"
                "Parameters\n"
                "----------\n"
                "position : array-like\n"
                "    Position vector in ECEF coordinates (meters)\n"
                "time : float\n"
                "    Time (typically in seconds since epoch)\n\n"
                "Returns\n"
                "-------\n"
                "float\n"
                "    Atmospheric density in kg/m³"
            );

        py::class_<AccelHarmonic>(m, "AccelHarmonic",
            "Spherical harmonic gravitational acceleration model.\n\n"
            "Computes gravitational acceleration using spherical harmonic expansion\n"
            "of the gravitational potential. This allows for high-fidelity modeling\n"
            "of non-uniform gravitational fields (e.g., Earth's J2, J3, J4 terms).\n\n"
            "Parameters\n"
            "----------\n"
            "GM : float\n"
            "    Gravitational parameter (m³/s²)\n"
            "R : float\n"
            "    Reference radius (meters)\n"
            "ncol : int\n"
            "    Number of columns in coefficient matrix\n"
            "CSptr : memory address\n"
            "    Pointer to spherical harmonic coefficients array\n\n"
            "Notes\n"
            "-----\n"
            "The coefficient matrix should contain normalized spherical harmonic\n"
            "coefficients Cnm and Snm arranged in the standard format."
        )
            .def(py::init(
                [](const double GM, const double R, const int ncol, size_t CSptr) {
                    return new AccelHarmonic(GM, R, ncol, reinterpret_cast<double*>(CSptr));
                }),
                "GM"_a, "R"_a, "ncol"_a, "CSptr"_a,
                "Initialize spherical harmonic acceleration model.\n\n"
                "Parameters\n"
                "----------\n"
                "GM : float\n"
                "    Gravitational parameter in m³/s²\n"
                "R : float\n"
                "    Reference radius in meters\n"
                "ncol : int\n"
                "    Number of columns in coefficient matrix\n"
                "CSptr : int\n"
                "    Memory address of coefficient array"
            )
            .def("accel",
                [](
                    const AccelHarmonic& ah,
                    const int n_max, const int m_max,
                    size_t inptr, size_t outptr
                ){
                    double* in = reinterpret_cast<double*>(inptr);
                    double* out = reinterpret_cast<double*>(outptr);
                    ah.accel(in[0], in[1], in[2], n_max, m_max, out[0], out[1], out[2]);
                },
                "n_max"_a, "m_max"_a, "inptr"_a, "outptr"_a,
                "Calculate gravitational acceleration using spherical harmonics.\n\n"
                "Computes acceleration vector at a given position using spherical\n"
                "harmonic expansion up to specified degree and order.\n\n"
                "Parameters\n"
                "----------\n"
                "n_max : int\n"
                "    Maximum degree of spherical harmonic expansion\n"
                "m_max : int\n"
                "    Maximum order of spherical harmonic expansion\n"
                "inptr : int\n"
                "    Memory address of input position vector [x, y, z]\n"
                "outptr : int\n"
                "    Memory address of output acceleration vector [ax, ay, az]\n\n"
                "Notes\n"
                "-----\n"
                "This is a low-level method. Use higher-level Python wrappers\n"
                "for more convenient access to gravitational acceleration calculations."
            );
    }

    PYBIND11_MODULE(_ssapy, m) {
        m.doc() = "SSAPy C++ Extension Module\n\n"
                  "High-performance orbital mechanics computations for the SSAPy\n"
                  "(Space Situational Awareness for Python) package.\n\n"
                  "This module provides optimized C++ implementations of:\n"
                  "- Ellipsoid coordinate transformations\n"
                  "- Atmospheric density models (Harris-Priester)\n"
                  "- Spherical harmonic gravitational acceleration";
        
        pyExportSSAPy(m);
    }
}