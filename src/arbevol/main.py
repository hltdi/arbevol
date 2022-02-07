#!/usr/bin/env python3

'''
arbevol. Evolution and learning of lexicons.
'''

from population import *

print("Welcome to arbevol, a program for investigating the learning and evolution of words.")

def init_pop(nmeanings=25, nhid=20, mlength=5, flength=5, noise=0.0, mvalues=5,
             iconic=True, dont_init=False, clusters=None, sep_amount=0.04,
             init_meanings=None, init_lex=None):
    pop = Population(10, iconic=iconic, nmeanings=nmeanings, nhid=nhid,
                     mlength=mlength, flength=flength, noise=noise, mvalues=mvalues,
                     clusters=clusters, dont_init=dont_init, sep_amount=sep_amount,
                     init_meanings=init_meanings, init_lex=init_lex)
    return pop

##def teach(population=None, tindex=0, sindex=1, popsize=10,
##          nmeanings=20, nhid=20, mlength=5, flength=4, noise=0.05, mvalues=4,
##          comptrain=0.5, prodtrain=0.5, iconic=True):
##    population = population or Population(popsize, iconic=iconic, nmeanings=nmeanings,
##                                          mlength=mlength, flength=flength, nhid=nhid,
##                                          noise=noise, mvalues=mvalues)
##    teacher = population[tindex]
##    student = population[sindex]
##    return teacher.teach(student, comptrain=comptrain, prodtrain=prodtrain)

#def teach2(population, tindex, sindex, comptrain=0.333, prodtrain=0.333):
#    teacher = population[tindex]
#    student = population[sindex]
#    return teacher.teach(student, comptrain=comptrain, prodtrain=prodtrain)

##def make_pops(nmeanings=12, nhid=8, mlength=4, flength=4, noise=0.05,
##              compprob=1.0, prodprob=0.0, mvalues=5):
##    iconpop = Population(2, iconic=True, nmeanings=nmeanings, nhid=nhid,
##                         mlength=mlength, flength=flength, noise=noise,
##                         compprob=compprob, prodprob=prodprob, mvalues=mvalues)
##    arbpop = Population(2, iconic=False, nmeanings=nmeanings, nhid=nhid,
##                        mlength=mlength, flength=flength, noise=noise,
##                        compprob=compprob, prodprob=prodprob, mvalues=mvalues)
##    return iconpop, arbpop
##
##def make_pops1():
##    '''
##    These values, trained for at least 5000 trials, reproduce the results of Gasser (2004).
##    ip, ap = make_pops1()
##    ie, ae = make_exp(iconpop=ip, arbpop=ap)
##    ie.run(10000)
##    ae.run(10000)
##    ie.test('comp')
##    ae.test('comp')
##    '''
##    return make_pops(nmeanings=12, nhid=8, mlength=4, flength=4, noise=0.05,
##                     compprob=1.0, prodprob=0.0, mvalues=5)


##def make_exp(nmeanings=50, nhid=10, mlength=5, flength=5, iconpop=None, arbpop=None,
##             noise=0.1, comptrain=1.0, prodtrain=0.0, mvalues=4):
##    if not iconpop:
##        iconpop, arbpop = make_pops(nmeanings=nmeanings, nhid=nhid,
##                                    mlength=mlength, flength=flength, noise=noise,
##                                    mvalues=mvalues)
##    iconteacher = iconpop[0]
##    iconstudent = iconpop[1]
##    arbteacher = arbpop[0]
##    arbstudent = arbpop[1]
##    print("I teacher {}, lexicon {}".format(iconteacher, iconteacher.lexicon))
##    print("A teacher {}, lexicon {}".format(arbteacher, arbteacher.lexicon))
##    iconexp = iconteacher.teach(iconstudent, comptrain=comptrain, prodtrain=prodtrain)
##    arbexp = arbteacher.teach(arbstudent, comptrain=comptrain, prodtrain=prodtrain)
##    return iconexp, arbexp

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Specify an experiment or a world file
        # Do graphics
        root = Tk()
        # Find an Experiment with the name passed as the first argument
        experiment = EXPERIMENTS.get(sys.argv[1], None)
        if not experiment:
            if len(EXPERIMENTS) > 0:
                experiment = list(EXPERIMENTS.values())[0]
        if experiment:
            experiment.display()
#    else:
#        print('You need to specify an experiment')
