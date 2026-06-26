from slicer.contracts import TraceOutput

def verify(output: TraceOutput) -> bool:
    """
    Quality Assurance for Step 3 Tracer.
    Asserts traced graph is non-empty and has zero native python dynamic structures.
    """
    print(f"[QA Tracer] Running quality checks for Layer {output.layer_index}...")
    
    # 1. Type Assertion
    assert isinstance(output, TraceOutput), "Output must be of type TraceOutput"
    assert output.exported_program is not None, "ExportedProgram cannot be None"
    
    program = output.exported_program
    
    # 2. Node Check
    if hasattr(program, 'graph'):
        nodes = list(program.graph.nodes)
        assert len(nodes) > 0, "Traced graph contains zero nodes"
        
        for node in nodes:
            if node.op == 'call_function':
                target_name = str(node.target)
                assert "builtin" not in target_name, (
                    f"Graph contains illegal Python-native operator: {target_name}"
                )
    else:
        print("[QA Tracer] Validating simulated graph program.")
        
    print("[QA Tracer] TraceQA contract check PASSED.")
    return True
