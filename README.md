i# Mental Gymnastics

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

When the episode starts this consists solely of the *source* and *sink* nodes.

### The Function Space

This is a dataset that represents a *function palette* which the agent may place into the Experiment Space and which contains:

* Input functions,
* Output functions,
* Atomic functions, and
* Composed functions

Input and Output functions are placed into the environment when the environment is instantiated; Atomic actions (when chosen) are used to create Composed functions.

The *size* of the palette is defined at runtime when the gym is created.

## The Action Space

The agent picks an action every turn by selecting a discrete value which is associated with one of the functions from the function bank.
The agent also chooses a continous location and a continuous radius.

In every time step the *state* is updated by inserting the function that was selected at the location and radius given.

*Atomic functions* placed will take all *non-sink* nodes within the *radius* and use those as input to create a new node at *location*.

*Composed functions* placed will *recreate* the composed function exactly.

## The Reward Function

There are a few reward functions available in the repository:

1. A *small* monotonic reward proportional to proximity to the sink.
2. A *slightly large* constant value of *C* which is rewarded if the agent has connected a node.
3. A *modest sized* constant value of *N* which is rewarded if the agent has connected from input to sink.

## Running Episodes, or an Experiment

In a single Experiment, an agent may run many trials. In a trial, or an episode, an agent will place *n* nodes before terminating the trial.
Limiting this to small numbers is likely to produce more generalizable results.

When an agent places a node it changes the in place net by adding a new node at that location and aggregating all the 'in radius nodes' in the manner specified by the action. If inputs can be simplified easily, they are (negotiable, this would add a layer of complexity, but would likely reduce computational burden.)

At the end of an episode the *net structure* that the agent created is saved as a single new action into the Action Space.


## Curating the Action Space

By default this stores all actions, though actions which have *gone stale* (i.e. haven't been in the leaderboard for some time and no layer in the leader board relies on them) will be pruned, and all their descendants will be removed as well.

This will help to limit the breadth of available actions.
