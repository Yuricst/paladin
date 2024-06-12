import os
import pygmo as pg
import pygmo_plugins_nonfree as ppnf


def run(library):
    uda = ppnf.snopt7(library=library, minor_version=7)
    algo = pg.algorithm(uda)
    udp = pg.ackley(50)
    prob = pg.problem(udp)
    pop = pg.population(prob, 1)
    pop = algo.evolve(pop)

    print("Done!")

if __name__=="__main__":
    run(os.getenv("SNOPT_SO"))

