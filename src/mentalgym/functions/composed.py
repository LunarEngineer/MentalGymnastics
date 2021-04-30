import json
import os
import numpy as np
import pandas as pd
import torch
from copy import deepcopy

from mentalgym.types import ExperimentSpace, FunctionBank
from typing import Any, Dict, Iterable, Optional, Union
import torch.nn as nn

class ComposedFunction():
    """Composed of multiple atomic functions.

    This class is used to build, run, and load a composed function.

    The 'id' is a string representing the *name* of the Function.
    The Composed Function will check the FunctionBank to
        see if an object with this id exists. If it does, then
        the Composed Function will attempt to load its weights
        from that location. If it is unable to load weights then
        an error is raised.
    If it *does not see a folder*, then this will build a model
        from a given Experiment Space.

    Parameters
    ----------
    experiment_space: Optional[ExperimentSpace]
        This is an ExperimentSpace object.
    function_bank: Optional[FunctionBank] = None
        This is a FunctionBank object.

    Methods
    -------
    forward(*args, **kwargs):
        The function signature of forward is such that it expects
        to see the inputs it was instantiated with when it is called
        with forward.

    Properties
    ----------

    """
    def __init__(
        self,
        id: str,
        experiment_space: Optional[ExperimentSpace] = None,
        function_bank: Optional[FunctionBank] = None,
        verbose: bool = True
    ):
        self._verbose = verbose
        # This property is storing the *locational* indices that the
        #   graph will expect to see the columns at.
        # i.e. {'input_0': 0, 'input_2': 1}
        # This can be unpacked later to appropriately feed information
        #   to the graph when forward is called.
        self.inputs = {}
        # 1) Is this building the net? We will check to see if the
        #   Function's directory exists.
        function_dir = os.path.join(
            function_bank._function_bank_directory,
            id
        )
        folder_exists = os.path.isdir(function_dir)
        # If the folder does not exist, then we are going to build
        #   the PyTorch graph for this net.
        
        self.module_dict = nn.ModuleDict()

        if not folder_exists:
            # Get a minimal subspace
            subspace = self.build_from_space(experiment_space)
            # Then turn that into a graph.
            self.build_forward(
                experiment_space = subspace,
                function_bank = function_bank
            )
            # And save that graph.
            self.save()
        # If the folder *does* exist, then we are going to load the
        #   graph for this net.
        else:
            self.load()
        # 
        # 
        # if has_bank:
        #     if has_space:
        #         self.model = self.build_from_space()
        #     else:
        #         self.model = self.load()
        # else:
        #     try:
        #         self.model = self.load()
        #     except:
        #         err_msg = """Composed Function Error:

        #         The Composed Function constructor was not given a
        #         FunctionBank object; when it attempted to pull a
        #         model from the given location it was unable to do so.

        #         Function Bank
        #         -------------\n{function_bank}

        #         Experiment Space
        #         ----------------\n{experiment_space}

        #         Function Directory
        #         ------------------\n{fn_path}
        #         """
        #         raise Exception(err_msg)

    def build_from_space(
        self,
        experiment_space: ExperimentSpace
    ) -> ExperimentSpace:
        """Create a minimal set of functions to build a graph.

        This, starting with the output node, will take an existing
        ExperimentSpace object and discard to only the required
        nodes which will be used to build the PyTorch computation
        graph.

        Parameters
        ----------
        experiment_space: ExperimentSpace
            This is a representation of Function objects.

        Returns
        -------
        minimal_experiment_space: ExperimentSpace
            This is a simplified representation of Function objects
            which will be used to construct a DaG.
        """
        # TODO: Ensure this is not changing the input dataframe.
        # This line ensures that all the 'No input' nodes have None
        #   values, instead of NaN. This is used elsewhere, where None
        #   values will trigger a recursion stop.
        experiment_space = experiment_space.fillna(
            np.nan
        ).replace([np.nan], [None])
        # Create new experiment space with only functions in the net.
        # This new space will have only intermediate and composite
        # functions in reverse order from output to input.
        net_df = pd.DataFrame().reindex(
            columns=experiment_space.columns
        )
        # This sets the first record of this space to be the *sink*
        #   node in the ExperimentSpace.
        net_df.loc[0] = experiment_space.query('type == "sink"').iloc[0]
        # Because the input is a *list* it must be recursively copied
        #   to prevent a 'pop' from affecting the primary dataset.
        cur_inputs = deepcopy(net_df.tail(1).input.item())
        # While the current inputs variable has values in it...
        while len(cur_inputs):
            # Snag the first one of those nodes and add it in to the
            #   output dataset.
            # cur_input = cur_inputs[0]
            cur_input = cur_inputs.pop(0)
            # But... only if it's not already in the output dataset.
            if cur_input in net_df.id.values:
                # If it already exists, skip it.
                continue
            # if cur_input not in net_df.id.values:
            # This sets the *final row* of the output dataset to
            #   the output of querying the ExperimentSpace for the
            #   cur_input 'id' value. This should always return a
            #   single value.
            net_df.loc[len(net_df.index)] = experiment_space.query(
                "id == @cur_input"
            ).iloc[0]
            # This, then gets the *new* inputs for that downstream
            #   node.
            inps = net_df.tail(1).input.item()
            # And each of these inputs will get added to the list
            #   of inputs that we're wandering down.
            if inps is not None:
                for inp in inps:
                    # TODO: How are we ensuring that we get the right
                    #   inputs for this? Is that handled in Christianne's
                    #   work?
                    # If it's not already in the output dataset and
                    #   it is *not* a source node, then add it to the
                    #   output dataset.
                    if (
                        inp not in net_df.id.values
                        and experiment_space.query(
                            "id == @inp"
                        ).type.values
                        != "source"
                    ):
                        cur_inputs.append(inp)
            # cur_inputs.pop(0)
        status_message = f"""Composed Function: build_from_space

        Input Experiment Space
        ----------------------\n{experiment_space}

        Output Experiment Space
        -----------------------\n{net_df}
        """
        if self._verbose:
            print(status_message)
        return net_df



    # def _recusive_init(
    #     self, 
    #     experiment_space: ExperimentSpace, 
    #     id: str
    # ):
    #     """ Populates the ModuleDict.

    #     Takes in a cleaned version of the experiment space.
    #     Recursively adds the modules to the ModuleDict, which is the
    #     equivalent of initiating those layers.

    #     """
    #     data = exp_space.query("id==@id")
    #     inputs = data.input.iloc[0]

    #     if inputs == None:
    #         return

    #     for ind in range(len(experiment_space)):
    #         fn_type = experiment_space.iloc[ind]['i']
    #         fn_id = experiment_space.iloc[ind]['id']
    #         fn_parameters = experiment_space.iloc[ind]['hyperparameters']

    #         if fn_type == relu_i:
    #             self.module_dict[fn_id] = nn.ReLU()
    #         elif fn_type == linear_i:
    #             self.module_dict[fn_id] = nn.Linear(
    #                     self.function_parameters["input_size"],
    #                     self.function_parameters["output_size"],
    #             )
    #         elif fn_type == dropout_i:
    #             self.module_dict[fn_id] = nn.Dropout(
    #                 self.function_parameters["p"]
    #             )
            
    #         self._recusive_init(experiment_space, ind)
        
    #     return 
            

    # def _recusive_forward(
    #     self, 
    #     experiment_space: ExperimentSpace, 
    #     id: str
    # ):
    #     """ Recursively calls forward on layers.

    #     Takes in the experiment space & ModuleDict.
    #     Recursively passes concatenated inputs to the next layer.
    #     Outputs the last layer's output.

    #     """
    #     data = exp_space.query("id==@id")
    #     inputs = data.input.iloc[0]
        
    #     type_ = data.type.iloc[0]       # get the type of the input we're currently on
    #     name = data.name.iloc[0]        # name of the input we're currently on

    #     output = torch.zeros(1)         # cannot concat empty tensors, so this must be zeros(1)

    #     if type_ == 'source':
    #         return torch.tensor(dataset.values[0], dtype=torch.float, requires_grad=True)  # return the modeling data point
        
    #     for inp in inputs:
    #         output = torch.cat((output, self._recusive_forward(experiment_space, id)))  # concatenate all the inputs
        
    #     output = output[1:]             # remove the added zeros(1) we created
        
    #     return self.module_dict[name](output)


    def build_forward(
        self,
        experiment_space: ExperimentSpace,
        function_bank: FunctionBank
    ):
        """Build a PyTorch Graph.

        When building a PyTorch net we generally start at the input.
        This graph is easiest to *construct* by starting at the output
          and walking backward. When we've walked backwards using a
          recursion function we have a completed net.
        
        This function is using the ExperimentSpace to do this. This
        does not return anything, but it sets the self.model and
        self.inputs properties.

        Parameters
        ----------
        experiment_space: ExperimentSpace
            This is an experiment space.
        """
        err_msg = """Build Forward Not Implemented:

        This function builds a PyTorch Graph from an experiment space.
        """
        raise NotImplementedError(err_msg)
        # This function is creating a ModuleDict to represent the
        # structure
        self._recusive_init(experiment_space, 'output')
        last_id = self._experiment_space.loc[
                          self._experiment_space["type"] == "sink", "input"
                      ].item()

        last_out = self._recusive_forward(experiment_space, last_id)   # last_id here because we haven't connected the output layer yet - nn.CrossEntropy or what have you

        # as a sanity check, last_out.requires_grad should be True...if it's not, then a comp graph wasn't built properly

        from torchviz import makedot

        makedot(last_out).render("comp_graph", format="png")    # visualize the output comp graph
        # --------------------------





        raise NotImplementedError(err_msg)

        # There is a recursor function and a recursive forward.
        # In the recursor you take the output and recursively build the input.
        # We are torch concatenating

    def save(self, repr, model):
        with open(os.path.join(self.fn_path, "connectivity_graph.json"), 'w') as f:
            f.write(json.dumps(repr))
        torch.save(model.state_dict(), os.path.join(self.fn_path, "weights.pt"))

    def load(self):
        # Load the net DF
        with open(os.path.join(self.fn_path, "connectivity_graph.json"), 'r') as f:
            self.repr = f.read()
        torch.load(os.path.join(self.fn_path, "weights.pt"))