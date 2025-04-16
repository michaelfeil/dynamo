from typer.testing import CliRunner
import os
from dynamo.sdk.cli.cli import cli
import re

runner = CliRunner()

def test_app():
    print(os.getcwd())
    result = runner.invoke(cli, ["serve", "pipeline:Frontend", "--working-dir", "deploy/dynamo/sdk/src/dynamo/sdk/tests", "--Frontend.model=qwentastic", "--Middle.bias=0.5", "--dry-run"])
    print(result.output)
    
    # Remove color codes printed by rich - TODO add a CI/Test flag that doesnt use rich
    clean_output = re.sub(r'\x1b\[[0-9;]*m', '', result.output)
    
    # Assert successful exit code
    assert result.exit_code == 0
    
    # With clean output we can do more precise checks
    assert 'Service Configuration:' in clean_output
    assert '"Frontend": {' in clean_output
    assert '"model": "qwentastic"' in clean_output
    assert '"Middle": {' in clean_output
    assert '"bias": 0.5' in clean_output
    assert 'Environment Variable that would be set:' in clean_output
    assert 'DYNAMO_SERVICE_CONFIG=' in clean_output
    assert '{"Frontend": {"model": "qwentastic"}, "Middle": {"bias": 0.5}}' in clean_output