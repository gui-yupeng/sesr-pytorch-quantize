import torch
from torch.fx import symbolic_trace
from torch.fx.node import Node

from myQL.quan_classes import NodeInsertMapping, NodeInsertConfig


def get_node_input(node: Node):
    return node.args


def set_node_input(node: Node, value):
    node.args = (value,)


def get_node_output(node: Node) -> Node:
    return node

def get_insert_config(node: Node, state_dict, node_insert_mapping: NodeInsertMapping) -> NodeInsertConfig:
    supported_operations = ['call_module']
    if node.op not in supported_operations:
        return NodeInsertConfig(should_insert=False)

    node_type = type(state_dict[node.target])

    node_mapping = node_insert_mapping.get_mapping()

    if node_type in node_mapping.keys():
        return NodeInsertConfig(should_insert=True, function_package=node_mapping[node_type])
    else:
        return NodeInsertConfig(should_insert=False)

def insert_before(model_input, insert_mapping: NodeInsertMapping, has_func_id = None) -> torch.fx.GraphModule:
    # Generate necessary components
    symbolic_traced_module = model_input
    if not isinstance(model_input, torch.fx.GraphModule):
        symbolic_traced_module = symbolic_trace(model_input)
    symbolic_traced_module_dict = dict(symbolic_traced_module.named_modules())
    symbolic_traced_module_graph = symbolic_traced_module.graph

    function_id = 0

    for current_node in symbolic_traced_module_graph.nodes:
        insert_config = get_insert_config(current_node, symbolic_traced_module_dict, insert_mapping)
        # If this node match the patter, a new node needs to be inserted after it
        if insert_config.should_insert:
            # Get the previous original node
            previous_origin_node = current_node.prev
            # Create temporary pointer for inserting
            with symbolic_traced_module_graph.inserting_before(current_node):
                # Create new node after current node
                func_args = insert_config.function_package.parameter_dict
                if has_func_id is not None:
                    func_args['func_id'] = function_id
                new_node = symbolic_traced_module_graph.call_function(insert_config.function_package.function,
                                                                      kwargs=func_args)
                function_id = function_id + 1
                # Set the input of the new node to the output of the previous original node
                set_node_input(new_node, get_node_output(previous_origin_node))
                # Get the output of the new node
                new_node_output = get_node_output(new_node)
                # Link the output of the new node to the input of the current node
                set_node_input(current_node, new_node_output)

    symbolic_traced_module_graph.lint()
    return torch.fx.GraphModule(model_input, symbolic_traced_module_graph)

def insert_bias_bypass(model_input, insert_mapping: NodeInsertMapping) -> torch.fx.GraphModule:
    # Generate necessary components
    model_state_dict = model_input.state_dict()
    symbolic_traced_module = model_input
    if not isinstance(model_input, torch.fx.GraphModule):
        symbolic_traced_module = symbolic_trace(model_input)
    symbolic_traced_module_dict = dict(symbolic_traced_module.named_modules())
    symbolic_traced_module_graph = symbolic_traced_module.graph

    function_id = 0

    latest_node_is_new_inserted = False
    for current_node in symbolic_traced_module_graph.nodes:
        # Skip an iteration if the last iteration inserts a new node, because this iteration is the new node
        if latest_node_is_new_inserted:
            # Next iteration will not enter this branch, it will be the originally existed node
            latest_node_is_new_inserted = False
        # Only originally existed node can Enter this branch.
        else:
            insert_config = get_insert_config(current_node, symbolic_traced_module_dict, insert_mapping)
            # If this node match the patter, a new node needs to be inserted after it
            if insert_config.should_insert:
                # Get the next original node
                next_origin_node = current_node.next
                # Create temporary pointer for inserting
                with symbolic_traced_module_graph.inserting_after(current_node):
                    # Cache the bias value of the current node
                    bias_value = model_state_dict[current_node.target + '.bias'].data

                    # Convert bias tensor to list
                    # ONLY LIST CAN BE PASSED TO "kwargs", AND SHOULD BE GENERATED BEFORE PASSED INTO IT
                    bias_list = bias_value.tolist()
                    
                    # Set bias of current node in state dict to 0
                    model_state_dict[current_node.target + '.bias'] = torch.zeros_like(bias_value)
                    # Load the new state dict
                    model_input.load_state_dict(model_state_dict)
                    parameters = insert_config.function_package.parameter_dict
                    parameters.update({'bias': bias_list})
                    parameters.update({'func_id':function_id})
                    # Create new node after current node
                    new_node = symbolic_traced_module_graph.call_function(insert_config.function_package.function,
                                                                          kwargs=parameters)
                    function_id = function_id + 1
                    # Set the input of the new node to the output of the current node
                    set_node_input(new_node, get_node_output(current_node))
                    # Get the output of the new node
                    new_node_output = get_node_output(new_node)
                    # Link the output of the new node to the input of the next original node
                    set_node_input(next_origin_node, new_node_output)

    symbolic_traced_module_graph.lint()
    return torch.fx.GraphModule(model_input, symbolic_traced_module_graph)


def insert_after(model_input, insert_mapping: NodeInsertMapping) -> torch.fx.GraphModule:
    # Generate necessary components
    symbolic_traced_module = model_input
    if not isinstance(model_input, torch.fx.GraphModule):
        symbolic_traced_module = symbolic_trace(model_input)
    symbolic_traced_module_dict = dict(symbolic_traced_module.named_modules())
    symbolic_traced_module_graph = symbolic_traced_module.graph
    function_id = 0

    for current_node in symbolic_traced_module_graph.nodes:
        insert_config = get_insert_config(current_node, symbolic_traced_module_dict, insert_mapping)
        # If this node match the patter, a new node needs to be inserted after it
        if insert_config.should_insert:
            # Get the next original node
            next_origin_node = current_node.next
            # Create temporary pointer for inserting
            with symbolic_traced_module_graph.inserting_after(current_node):
                # Create new node after current node
                func_args = insert_config.function_package.parameter_dict
                func_args['func_id'] = function_id
                new_node = symbolic_traced_module_graph.call_function(insert_config.function_package.function,
                                                                      kwargs=insert_config.function_package.parameter_dict)
                function_id = function_id + 1
                # Set the input of the new node to the output of the current node
                set_node_input(new_node, get_node_output(current_node))
                # Get the output of the new node
                new_node_output = get_node_output(new_node)
                # Link the output of the new node to the input of the next original node
                set_node_input(next_origin_node, new_node_output)

    symbolic_traced_module_graph.lint()
    return torch.fx.GraphModule(model_input, symbolic_traced_module_graph)
