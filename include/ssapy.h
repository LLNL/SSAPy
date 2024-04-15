#ifndef ssapy_ssapy_H
#define ssapy_ssapy_H

#include <memory>
#include <cmath>
#include <array>
#include <limits>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ssapy {

    /////////////////////////////////////////////////
    //
    // Ellipsoid
    //
    //   Class to handle transformations between ECEF x,y,z coords and geodetic
    //   longitude, latitude, and height.  Technically, only handles a one-axis
    //   ellipsoid, defined via a flattening parameter f, but that's good
    //   enough for simple Earth models.
    //
    //
    /////////////////////////////////////////////////
    class Ellipsoid {
    public:
        ////////////////////////////////////////////
        //
        // Constructor
        //
        // Parameters
        //
        //   Req : Earth radius at the equator.
        //   f : flattening parameter defined as (a-b)/a, where a and b are the
        //       semimajor and semiminor axes of the ellipse containing the Earth's
        //       rotational axis.
        //
        ////////////////////////////////////////////
        Ellipsoid(const double Req=6378.137e3, const double f=1.0/298.257223563) :
            _Req(Req), _f(f), _e2(f*(2.0-f)), _1mf2((1-f)*(1-f)) {}

        ////////////////////////////////////////////
        //
        // sphereToCart
        //
        //   Convert geodetic "spherical" coordinates to cartesian coordinates.
        //
        // Parameters
        //  lon      [in] : Longitude in radians
        //  lat      [in] : Latitude in radians
        //  height   [in] : Height above ellipsoid
        //  x, y, z [out] : Cartesian coordinates
        //
        ////////////////////////////////////////////
        void sphereToCart(
            const double lon, const double lat, const double height,
            double& x, double& y, double& z
        ) const;

        ////////////////////////////////////////////
        //
        // cartToSphere
        //
        //   Convert cartesian coordinates to geodetic "spherical" coordinates.
        //
        // Parameters
        //  x, y, z [in] : Cartesian coordinates
        //  lon    [out] : Longitude in radians
        //  lat    [out] : Latitude in radians
        //  height [out] : Height above ellipsoid
        //
        ////////////////////////////////////////////
        void cartToSphere(
            const double x, const double y, const double z,
            double& lon, double& lat, double& height
        ) const;

    private:
        const double _Req;
        const double _f;
        const double _e2;  // e^2 = f*(2-f)  MG (5.86)
        const double _1mf2;  // (1-f)^2
    };


    /////////////////////////////////////////////////
    //
    // HarrisPriester
    //
    //   Atmospheric density model of Harris and Priester (1962)
    //
    /////////////////////////////////////////////////
    class HarrisPriester {
    public:
        ////////////////////////////////////////////
        //
        // Constructor
        //
        // Parameters
        //
        //   ellip : Earth ellipsoid.
        //   n     : A scaling parameter for inclination.  Low inclination
        //           orbits have n ~ 2, polar orbits have n ~ 6.
        //
        ////////////////////////////////////////////
        HarrisPriester(
            const Ellipsoid& ellip, double n=6.0
        ) : _ellip(ellip), _n(n) {}

        ////////////////////////////////////////////
        //
        // density
        //
        //   Compute atmospheric density
        //
        // Parameters
        //  x, y, z [in] : True-of-date Cartesian equatorial coordinates in meters.
        //  ra_Sun  [in] : True-of-date right ascention of sun in radians.
        //  dec_Sun [in] : True-of-date declination of sun in radians.
        //
        // Returns
        //  density in kg/m^3
        ////////////////////////////////////////////
        double density(
            const double x, const double y, const double z,
            const double ra_Sun, const double dec_Sun
        ) const;


    private:
        const double _n;
        const Ellipsoid& _ellip;

        // MG Table 3.8
        static const std::array<double, 50> _h;
        static const std::array<double, 50> _rho_antapex;  // density at bulge antipode
        static const std::array<double, 50> _rho_apex;  // density at bulge
        static const std::array<double, 49> _scale_antapex;
        static const std::array<double, 49> _scale_apex;

        static const std::array<double, 49> _compute_scale_antapex();
        static const std::array<double, 49> _compute_scale_apex();
    };


    // Super simple matrix class just to enable mat(i,j) element access.
    class Mat {
    public:
        Mat(const int rows, const int cols) :
            _owns(true), _ncol(cols)
        {
            _data = new double[rows*cols];
            for (int i = 0; i<(rows*cols); i++) {
                _data[i] = 0;
            }
        }

        Mat(const int ncol, double* const data) :
            _owns(false), _ncol(ncol), _data(data)
        {}

        ~Mat() {
            if(_owns) delete _data;
        }

        double& operator()(int irow, int icol) {
            return _data[icol + _ncol*irow];
        }

        double& operator()(int irow, int icol) const {
            return _data[icol + _ncol*irow];
        }

    private:
        const bool _owns;
        const int _ncol;
        double* _data;
    };


    /////////////////////////////////////////////////
    //
    // AccelHarmonic
    //
    //   Compute acceleration from harmonic expansion of gravitational
    //   field.
    //
    /////////////////////////////////////////////////
    class AccelHarmonic {
    public:
        ////////////////////////////////////////////
        //
        // Constructor
        //
        // Parameters
        //
        //   GM   : Gravitational parameter in m^3/s^2
        //   R    : Reference radius in m
        //   ncol : Number of columns in CS array
        //   CS   : Harmonic coefficients stored as
        //          C[n,m] = CS[n,m]
        //          S[n,m] = CS[m-1,n]
        //
        ////////////////////////////////////////////
        AccelHarmonic(
            const double GM, const double R,
            const int ncol, double* const CSptr
        ) :
            _R(R), _Rsq(R*R), _GMinvRsq(GM/_Rsq), _ncol(ncol),
            _CS(_ncol, CSptr), _V(_ncol+1, _ncol+1), _W(_ncol+1, _ncol+1)
        {}

        ////////////////////////////////////////////
        //
        // accel
        //
        //   Compute graviational acceleration
        //
        // Parameters
        //  x, y, z     [in] : True-of-date Cartesian equatorial coordinates in meters.
        //  n_max       [in] : Maximum order of model to compute.
        //  m_max       [in] : Maximum degree of model to compute.
        //  ax, ay, az [out] : Acceleration in m/s^2.
        //
        ////////////////////////////////////////////
        void accel(
            const double x, const double y, const double z,
            const int n_max, const int m_max,
            double& ax, double& ay, double& az
        ) const;

    private:
        const double _R;
        const double _Rsq; // R_ref^2
        const double _GMinvRsq; // GM / R_ref^2
        const int _ncol;
        Mat _CS;  // Harmonic coefficients
        Mat _V, _W;  // Preallocate scratch arrays.
    };
}

#endif
