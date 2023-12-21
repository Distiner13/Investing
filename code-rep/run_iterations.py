# run_iterations.py

import subprocess

def run_script(script_path, iterations, parameter_values):
    for param_value in parameter_values:
        # Read the template script
        with open(script_path, 'r') as file:
            script_content = file.read()

        # Replace the placeholder with the actual parameter value
        modified_script = script_content.replace("default_parameter = 10", f"default_parameter = {param_value}")

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
    #make it set the name of the paramter with the range of values instead of the number of iterations
    iterations = 5
    parameter_values = [1, 2, 3, 4, 5]  # Add your parameter values

    run_script(script_path, iterations, parameter_values)
