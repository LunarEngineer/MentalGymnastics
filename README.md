# Mental Gymnastics

This is an OpenAI compatible reinforcement learning environment to deliver the final project for Deep Learning, Spring 2021.

This is an environment where you can specify a number of atomic *functions* (with code) that you wish an agent to consider placing into an experiment.

The agent, beginning with these atomic functions and an input dataset, will begin experimenting and will add higher level functions into it's function space as it explores and evolves to solve a defined problem.

## The State Space

The environment presents a complex state space composed of:

* The Experiment Space,
* The Function Space,
* Metrics associated with the Function Space.

### The Experiment Space

This is a dataset of experiment nodes. This is an *episode length* long iterable where each element contains:

* A *function id* for the functions which have been added to the Experiment Space,
* A *location* at which the *function node* exists at in this Experiment Space.

### The Function Space

This is a dataset that represents a *function palette* which the agent may place into the Experiment Space and which contains:

* Input functions,
* Output functions,
* Atomic functions, and
* Composed functions

Input and Output functions are placed into the environment when the environment is instantiated; Atomic actions (when chosen) are used to create Composed functions.

## The Action Space

## The Reward Function

The reward system is linearly superimposed with an optional reward function that *you write*, which can access Experiment Space and Action space when calculating.
A default reward system is embedded which rewards the agent with logarithmically increasing reward (negotiable) based on reduced variance in predictions.

Additionally, the agent is *only* rewarded if it places an action which does not already exist in the space in the same location; if no changes are made to the DAG, no rewards are earned.

Within the environment the bank of available actions draws the top *m* (based on score metric) and random function *m* actions from previously defined actions (uniform random by default).
These previously defined actions are *created* by the agent.

*A conversation about how to create and store the layers is in order.

## Running Episodes, or an Experiment

In a single Experiment, an agent may run many trials. In a trial, or an episode, an agent will place *n* nodes before terminating the trial.
Limiting this to small numbers is likely to produce more generalizable results.

When an agent places a node it changes the in place net by adding a new node at that location and aggregating all the 'in radius nodes' in the manner specified by the action. If inputs can be simplified easily, they are (negotiable, this would add a layer of complexity, but would likely reduce computational burden.)

At the end of an episode the *net structure* that the agent created is saved as a single new action into the Action Space.


## Curating the Action Space

By default this stores all actions, though actions which have *gone stale* (i.e. haven't been in the leaderboard for some time and no layer in the leader board relies on them) will be pruned, and all their descendants will be removed as well.

This will help to limit the breadth of available actions.
