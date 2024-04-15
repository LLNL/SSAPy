// Script used to unit test proper motion, parallax, and aberration corrections in SSA library.

#include <stdio.h>
#include "sofa.h"

void reprd ( char*, double, double );

int main () {
    iauASTROM astrom;
    double utc1, utc2, tai1, tai2, tt1, tt2,
           rc, dc, pr, pd, px, rv,
           eo, ri, di, rca, dca;
/* UTC date. */
    if ( iauDtf2d ( "UTC", 2013, 4, 2, 23, 15, 43.55,
                   &utc1, &utc2 ) ) return -1;
/* TT date. */
    if ( iauUtctai ( utc1, utc2, &tai1, &tai2 ) ) return -1;
    if ( iauTaitt ( tai1, tai2, &tt1, &tt2 ) ) return -1;
/* Star ICRS RA,Dec (radians). */
    if ( iauTf2a ( ' ', 14, 34, 16.81183, &rc ) ) return -1;
    if ( iauAf2a ( '-', 12, 31, 10.3965, &dc ) ) return -1;
    reprd ( "ICRS, epoch J2000.0:", rc, dc );
/* Proper motion */
    pr = atan2 ( -354.45e-3 * DAS2R, cos(dc) );
    pd = 595.35e-3 * DAS2R;
    px = 0.0;
    rv = 0.0;
    iauAtci13 ( rc, dc, pr, pd, px, rv, tt1, tt2, &ri, &di, &eo );
    iauAtic13 ( ri, di, tt1, tt2, &rca, &dca, &eo );
    reprd ( "Try proper motion:", rca, dca );
/* Parallax */
    pr = 0.0;
    pd = 0.0;
    px = 0.16499;
    rv = 0.0;
    iauAtci13 ( rc, dc, pr, pd, px, rv, tt1, tt2, &ri, &di, &eo );
    iauAtic13 ( ri, di, tt1, tt2, &rca, &dca, &eo );
    reprd ( "Try parallax:", rca, dca );
/* aberration */
    pr = 0.0;
    pd = 0.0;
    px = 0.0;
    rv = 0.0;
    double pco[3], ppr[3], rg, dg, w;
    iauApcg13(tt1, tt2, &astrom);
    iauPmpx(rc, dc, pr, pd, px, rv, astrom.pmt, astrom.eb, pco);
    iauAb(pco, astrom.v, astrom.em, astrom.bm1, ppr);
    iauC2s(ppr, &w, &dg);
    rg = iauAnp(w);
    reprd ( "aberration:", rg, dg);
/* all of above */
    pr = atan2 ( -354.45e-3 * DAS2R, cos(dc) );
    pd = 595.35e-3 * DAS2R;
    px = 0.16499;
    rv = 0.0;
    iauApcg13(tt1, tt2, &astrom);
    iauPmpx(rc, dc, pr, pd, px, rv, astrom.pmt, astrom.eb, pco);
    iauAb(pco, astrom.v, astrom.em, astrom.bm1, ppr);
    iauC2s(ppr, &w, &dg);
    rg = iauAnp(w);
    reprd ( "all above:", rg, dg);

    return 0;
}

void reprd ( char* s, double ra, double dc )
{
    char pm;
    int i[4];
    printf ( "%25s", s );
    iauA2tf ( 7, ra, &pm, i );
    printf ( " %2.2d %2.2d %2.2d.%7.7d", i[0],i[1],i[2],i[3] );
    iauA2af ( 6, dc, &pm, i );
    printf ( " %c%2.2d %2.2d %2.2d.%6.6d\n", pm, i[0],i[1],i[2],i[3] );
}
