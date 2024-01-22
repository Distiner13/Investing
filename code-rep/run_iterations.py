'''
This was meant to facilitate the repition process of contniously running a single file manually,
you can pretty much use this to run the same program multiple times using a different value
for instance, you can run the LLS program wiyth different stock prices manually or modify it to 
use this program below and automate the repetition (incomplete and needs to make modifications to all other programs)

(NOT Functional, for now)
'''


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
