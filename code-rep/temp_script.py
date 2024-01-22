'''
This is what other programs should have around them if they want to be compatible
with 'run_iterations.py', modify this program to fit other shit
'''


#%%
# template_script.py

def main(parameter):
    # Your existing code here that uses the 'parameter' variable
    result = some_function(parameter)
    print(f"Parameter: {parameter}, Result: {result}")

def some_function(param):
    # Your existing function implementation
    return param * 2

if __name__ == "__main__":
    # You can leave this block as is or modify as needed
    default_parameter = 5
    main(default_parameter)


