
import pygmo as pg
import pygmo_plugins_nonfree as ppnf


def algo_gradient(
    name="ipopt", 
    ftol=1e-4,
    ctol=1e-6,
    original_screen_output=False,
    snopt7_path="/usr/local/lib/libsnopt7_cpp.so",
    max_iter=1000,
    minor_version=7,
):
    """Get gradient based algorithm"""
    if name == "slsqp":
        uda = pg.nlopt('slsqp')
        uda.xtol_rel = 1e-5
        uda.ftol_rel = 0
        algo = pg.algorithm(uda)
        algo.set_verbosity(1)

    elif name == "ipopt":
        if original_screen_output:
            pl = 5
        else:
            pl = 0
        # Disable lint check on next line. Known issue (pagmo2/issues/261)
        uda = pg.ipopt() # pylint: disable=no-member
        uda.set_integer_option("print_level", pl)
        uda.set_integer_option("acceptable_iter", 4)
        uda.set_integer_option("max_iter", max_iter)

        uda.set_numeric_option("tol", ftol)
        uda.set_numeric_option("dual_inf_tol", 1e-6)
        uda.set_numeric_option("constr_viol_tol", ctol)
        uda.set_numeric_option("compl_inf_tol", 1e-6)

        uda.set_numeric_option("acceptable_tol", 1e-3)
        uda.set_numeric_option("acceptable_dual_inf_tol", 1e-2)
        uda.set_numeric_option("acceptable_constr_viol_tol", 1e-6)
        uda.set_numeric_option("acceptable_compl_inf_tol", 1e-6)
        algo = pg.algorithm(uda)

    elif name == "snopt" or name == "snopt7":
        uda = ppnf.snopt7(library=snopt7_path, minor_version=minor_version) 
        #(original_screen_output, snopt7_path)
        uda.set_integer_option("Major iterations limit", max_iter)
        uda.set_integer_option("Iterations limit", 200000)
        uda.set_numeric_option("Major optimality tolerance", 1e-2)
        uda.set_numeric_option("Major feasibility tolerance", ftol)
        uda.set_numeric_option('Major feasibility tolerance', ctol)
        uda.set_numeric_option('Minor feasibility tolerance', ctol)
        algo = pg.algorithm(uda)

    return algo