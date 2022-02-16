# Parameters
## Meanings
* Clusters: `clusters=(prop, nclusters, mperclus)` or `None` for arbitrary meanings
   * `prop`: the proportion of meaning dimensions fixed for each cluster
	* `nclusters`: number of clusters
	* `nperclus`: number of meanings per cluster
* Number of dimensions: `mlength`
* Number of values: `mvalues`

## Forms

* Number of dimensions: `flength`
* Number of values (same as for meanings): `mvalues`
* Whether iconicity is built into initial lexicon: `iconic`
	* Proportion of dimensions that are not copied from meaning in initial lexicon: `deiconize` (actual number copied also depends on `mvalues`
	* Whether the non-copied dimensions are the same for all entries: `constant_flip`
* Whether features in all forms are rounded to the nearest of `mvalues` values (0<=v<=1): `Form.roundQ`.

	
## Training and networks
* Network hidden layer size: `nhid`
* Number of training "cycles" (in `meta_experiment`): `nruns`
* What happens during the initialization cycle.
	* The "master teacher" creates an initial `Lexicon`, based on parameters controlling `Meaning`s and `Form`s.
	* The master teacher trains its comprehension, production, and joint networks, using `teach` with teacher and student the same, updating its `Lexicon` on the basis of the form layer of the joint network.
	* Until each other `Person` has been trained on a `Lexicon`, a `Person` already trained is selected randomly to train an untrained `Person`. The student's comprehension, production, and joint networks are trained in the target `Lexicon`, using `teach`, and its initial `Lexicon` is created from the form layer of its joint network.
* What happens on a training cycle. For randomly selected pair of `Person`s, `p1` and `p2`,
	* If `match` is `True`, run `match_meanings` on the pair, updating `p1`'s lexicon if there are misses, that is, if the meaning input to `p1'`s production network does not match the output of `p2`'s comprehension network, given the output of `p1`'s network as input. After updates, retrain `p1`'s comprehension, production, and joint networks.
	* If `match_meanings` found any misses and `teach` is `True` or if `match` is `False`, run `teach` on the pair, with `p1` as the teacher. Train `p2`'s comprehension, production, and joint networks on entries in `p1`'s current lexicon, updating `p2`'s lexicon based on the forms in its joint network.
* Parameters controlling `teach`
	* How many trials to train comprehension and production networks (`trials_per_lex`) and joint network (`joint_trials_per_lex`) [Note: this may vary for the first and subsequent calls to `teach` and for training of the "master teacher" on its initial `Lexicon` vs. training of non-teachers by teachers during initialization.]
	* What learning rates to use for training comprehension and production networks (`lr`) and joint network (`joint_lr`) [Note: this may vary for the first and subsequent calls to `teach`.]
	* Whether noise is added to input and target forms: `noise`. If `noise` is not 0.0, its value is used for the `sd` parameter in `noisify`.
* Parameters controlling `match_meanings`
	* How many trials to train comprehension and production networks (`trials_per_lex`) and joint network (`joint_trials_per_lex`) of "learner"
	* How many times to retrain "learner" if there are still missed meanings (but fewer than on the last repetition): `nreps`
* How lexicons are updated
	* `test_all` run on a `Person`'s joint network or the combined production and comprehension networks of two `Person`s during `match_meanings` saves the forms from the appropiate layer of the network; these are then used to update the lexicon of the "learner". `test_all` also records "misses", meanings which are not correctly reproduced on the output of the comprehension network. During updating, the forms for these entries are modified.
* Parameters controlling how missed forms are changed
	* Whether noise is added vs. separating current form from form associated with incorrect output meaning: `noisify_errors`
	* For `separate`, how far to move a form from the form it's confused with: `sep_amount`.

# Measures
* Distance: distance between corresponding entries in lexicons of two `Person`s.
* Mutuality: network output error and miss error for pair of `Person`s tested using `test_all`.
* Iconicity: entire lexicon, within clusters, between clusters
	* For each form dimension, highest absolute correlation with a meaning dimension (`iconicity`)
	* For each form dimension, correlation with each meaning dimension (`dim_iconicity`)
	* Correlation of distances between pairs of meanings and their corresponding forms (`distcorr`)

# Some basic results
1. All `Person`s quickly converge on the same `Lexicon`s. Population size, at least between 5 and 10, doesn't seem to matter much.
2. There is great variation from one run (new `Population`) to another, given the same set of parameters settings, apparently due both to spurious structure in the initial `Lexicon` as well as the initial weights of the "master teacher".
2. Different ways of fixing errors (forms that result in missed output meanings of joint and paired networks) have little effect on overall and dimension-specific iconicity, however measured.
3. Structured meaning spaces (with clusters) are overall more difficult to learn than random spaces, but the expected lower iconicity does not appear consistently under any conditions.
4. In difficult situations, for whatever reason, the networks may not converge on a better solution after many training cycles, even when this is apparently possible given the current resources.
5. There's some indication that some form dimensions may specialize in particular meaning dimensions, in particular one of the within-cluster constant dimensions. But even when there's redundancy among these dimensions, more than one may end up attended to.