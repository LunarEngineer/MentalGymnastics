import json
import os
import numpy as np
import pandas as pd
import torch
from copy import deepcopy

from torch.nn.modules.container import ModuleDict

from mentalgym.functions import atomic_constants
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
        self._n_inputs = 0
        self.input = {}
        self._net_subspace = None
        self._verbose = verbose
        self._module_dict = None
        self._function_bank = None
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

        if not folder_exists:
            # Get a minimal subspace
            self.build_from_space(experiment_space)
            # Then turn that into a graph.
            self.build_forward(
                experiment_space = experiment_space,
                function_bank = function_bank
            )
            # And save that graph.
            self.save()
        # If the folder *does* exist, then we are going to load the
        #   graph for this net.
        else:
            self.load()

    def _recusive_init(
        self, 
        id: str
    ):
        """Populates a ModuleDict with Atomic and Composed functions.

        This uses a minimal subset of information to recursively add
        Atomic and Composed functions to the ModuleDict; this is
        equivalent to calling these layers in an __init__ function.

        Parameters
        ----------
        id: str
            This is the *name* of the layer.
        """
        # If this is your first night at Fight Club, you have to fight.
        if self._module_dict is None:
            self._module_dict = ModuleDict()
        # This fetches just the row for this function.
        # The first time this is called it gets the *sink* row.
        print(f"MY DATA IS {self._net_subspace}")
        print("MY QUERY IS :", id)
        data = self._net_subspace.query("id==@id")
        print(f"MY FILTERED DATA IS {data}")
        # This, then, gets the list of input nodes for this function.
        inputs = data.input.iloc[0]
        print("MY INPUTS ARE: ", inputs)

        # If the inputs are None, then this ceases recursion, because
        #   this is a *source* node.
        if inputs == None:
            self._data
            return

        # If the inputs are *not* None, then this has input!
        # For every input node to connect...
        for ind in inputs:
            ########################################################
            #       Instantiate the Input Nodes for this Layer     #
            ########################################################
            # Get the row for this input node:
            row = self._net_subspace.query("id==@ind")
            # Pull out the ID; we're going to use this ID as the key
            #   in the ModuleDict.
            fn_id = row['id'].item()
            # Then pull out the hyperparameters
            fn_parameters = row['hyperparameters'].item()
            # And pull out the Object.
            fn_object = row['object'].item()
            status_message = f"""Build Composed Layer Init:

            ########################################################
            # Function Information                                 #
            ########################################################

            Function ID:         {fn_id}
            Function Parameters: {fn_parameters}
            Function Object:     {fn_object}
            """
            if self._verbose:
                print(status_message)
            # And instantiate the object!
            self.module_dict[fn_id] = fn_object(**fn_parameters)
            # Now, call recursive init on *this* input to walk down
            #   the computation tree.
            self._recusive_init(id = ind)
            ########################################################
            # Holdover Code: Do not delete until the above is tested.
            # self.module_dict[fn_id] = atomic_constants[fn_i]
            # if fn_type == relu_i:
                
            # elif fn_type == linear_i:
            #     self.module_dict[fn_id] = nn.Linear(
            #             self.function_parameters["input_size"],
            #             self.function_parameters["output_size"],
            #     )
            # elif fn_type == dropout_i:
            #     self.module_dict[fn_id] = nn.Dropout(
            #         self.function_parameters["p"]
            #     )
        return

    def _recusive_forward(
        self,
        id: str,
        x: torch.Tensor
    ) -> nn.Module:
        """Recursively call forward on internal layers.

        This uses the internal minimal experiment subspace and the
        module dictionary to call the forward. This recursively
        passes concatenated inputs to the next layer. This forward
        outputs the last layer's (prior to the sink) output.

        Parameters
        ----------
        id: str
            The ID of the layer in the experiment space.
        x: torch.Tensor
            The input data which is getting forward called on it.
            This is an m x len(self.inputs) tensor with each of the
            columns representing one of the inputs which goes into
            this composed function. This can be used to pull out
            required data from a dataset.
        """
        # Pull out the row for this ID
        data = self._net_subspace.query("id==@id")
        # Then get the inputs that feed into it.
        inputs = data.input.iloc[0]
        # And determine if it is a source node.
        is_source = data.input.iloc[0] == 'source'
        # If it's a source node we're simply going to return the
        #   column of the dataset that matches *this* input.
        if is_source:
            # This points at the correct input in the passed tensor.
            # This is assuming Torch input.
            input_col: torch.Tensor = map_to_output(
                x,
                self.inputs[id]
            )
            #
            layer = torch.tensor(
                # TODO: Fix this to point at the right column
                input_col,
                dtype = torch.float, # TODO: This needs to be handled differently, I think.
                requires_grad = True
            )
            return layer
        # If it's *not* a source, then we recurse.
        for inp in inputs:
            output = torch.cat(
                (
                    output,
                    self._recusive_forward(id, x)
                )
            )  # concatenate all the inputs
        
        type_ = data.type.iloc[0]       # get the type of the input we're currently on
        name = data.name.iloc[0]        # name of the input we're currently on

        output = torch.zeros(1)         # cannot concat empty tensors, so this must be zeros(1)

        if type_ == 'source':
            return torch.tensor(dataset.values[0], dtype=torch.float, requires_grad=True)  # return the modeling data point
        
        
        
        output = output[1:]             # remove the added zeros(1) we created
        
        return self.module_dict[name](output)


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
        err_msg = """Build Forward Not Fully Implemented:

        This function builds a PyTorch Graph from an experiment space.
        """
        # This function is creating a ModuleDict to represent the
        # structure
        # Get the id of the sink:
        sink_id = self._net_subspace.query('type=="sink"').id.item()
        # TODO: Working in recursive init right now.
        self._recusive_init(sink_id)
        raise Exception(err_msg)
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

    def build_from_space(
        self,
        experiment_space: ExperimentSpace
    ) -> ExperimentSpace:
        """Create a minimal set of functions to build a graph.

        This, starting with the output node, will take an existing
        ExperimentSpace object and discard to only the required
        nodes which will be used to build the PyTorch computation
        graph. That will be stored as a class property.

        Parameters
        ----------
        experiment_space: ExperimentSpace
            This is a representation of Function objects.

        Returns
        -------
        minimal_experiment_space: ExperimentSpace
            This is a simplified representation of Function objects
            which will be used to construct a DaG. This downselects
            to *only* the information that is required to build the
            graph:
              * i
              * id
              * type
              * input
              * hyperparameters
              * object
        """
        persist_fields = [
            'i', 'id', 'type', 'input', 'hyperparameters', 'object'
        ]
        # TODO: Ensure this is not changing the input dataframe.
        # This line ensures that all the 'No input' nodes have None
        #   values, instead of NaN. This is used elsewhere, where None
        #   values will trigger a recursion stop.
        err_msg = """Composed Function Error:

        The Composed Function constructor was not given an
        ExperimentSpace object.
        """
        assert experiment_space is not None, err_msg
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
                # If it already exists, skip this iteration.
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
        net_df = net_df[persist_fields]
        status_message = f"""Composed Function: build_from_space

        Input Experiment Space
        ----------------------\n{experiment_space}

        Output Minimal Experiment Space
        -------------------------------\n{net_df}
        """
        if self._verbose:
            print(status_message)
        # Now, assign the property.
        self._net_subspace = net_df


def map_to_output(
    x: torch.tensor,
    inputs: Dict[str, int]
):
    """Subsets a tensor to provide the correct input data.

    Using a dict like {'input_0': 6, 'input_1': 3} put together
    an m x 2 dataset from a torch tensor.
    """
    err_msg = """TODO:
    Use a key mapping to pull back appropriate input.
    """
    raise NotImplementedError