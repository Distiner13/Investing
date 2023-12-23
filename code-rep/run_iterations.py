#%%
# run_iterations.py

import subprocess

def run_script(script_path, parameter, parameter_values):
    for param_value in parameter_values:
        # Read the template script
        with open(script_path, 'r') as file:
            script_content = file.read()

        # Replace the placeholder with the actual parameter value
        modified_script = script_content.replace(f"{parameter} = 10", f"default_parameter = {param_value}")

        # Write the modified script to a temporary file
        with open('temp_script.py', 'w') as temp_file:
            temp_file.write(modified_script)

        # Run the modified script
        command = ['python', 'temp_script.py']
        subprocess.run(command)

if __name__ == "__main__":
    # Replace 'template_script.py' with the name of your template script
    script_path = 'template_script.py'

    # Set the number of iterations and the range of parameter values
    parameter = "default_parameter"
    parameter_values = [1, 2, 3, 4, 5]  # Add your parameter values

    run_script(script_path, parameter, parameter_values)
