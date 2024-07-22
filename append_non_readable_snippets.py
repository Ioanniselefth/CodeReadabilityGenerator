import pandas as pd

# Load the existing CSV file
df = pd.read_csv('mnt/data/code_snippets.csv')

# Examples of non-readable code snippets
non_readable_snippets = [
    {
        'problem_title': 'Poorly Indented Python Code',
        'python_solutions': 'def calculate_area(radius):\nreturn 3.14 * radius * radius',
        'difficulty': 1,
        'num_of_lines': 2,
        'code_length': 48,
        'comments': 0,
        'cyclomatic_complexity': 1,
        'indents': 0,
        'loop_count': 0,
        'line_length': 48,
        'identifiers': 3,
        'readability': 0  # Assuming readability score is from 0 (not readable) to 5 (very readable)
    },
    {
        'problem_title': 'Random Text',
        'python_solutions': '1234',
        'difficulty': 1,
        'num_of_lines': 1,
        'code_length': 4,
        'comments': 0,
        'cyclomatic_complexity': 0,
        'indents': 0,
        'loop_count': 0,
        'line_length': 4,
        'identifiers': 0,
        'readability': 0
    },
    {
        'problem_title': 'Poorly Formatted SQL',
        'python_solutions': 'SELECT name,age FROM users WHERE age>20AND city=\'New York\';',
        'difficulty': 1,
        'num_of_lines': 1,
        'code_length': 58,
        'comments': 0,
        'cyclomatic_complexity': 0,
        'indents': 0,
        'loop_count': 0,
        'line_length': 58,
        'identifiers': 6,
        'readability': 0
    },
    {
        'problem_title': 'JavaScript with Syntax Errors',
        'python_solutions': 'function checkAge(age {\nif age < 18 {\nconsole.log("Minor");\n} else {\nconsole.log("Adult");\n}\n}',
        'difficulty': 1,
        'num_of_lines': 6,
        'code_length': 81,
        'comments': 0,
        'cyclomatic_complexity': 2,
        'indents': 0,
        'loop_count': 0,
        'line_length': 81,
        'identifiers': 7,
        'readability': 0
    },
    {
        'problem_title': 'HTML with Missing Tags',
        'python_solutions': '<html>\n<head><title>Sample Page</title></head>\n<body><h1>Title</h1><p>Content here\n</body>\n</html>',
        'difficulty': 1,
        'num_of_lines': 5,
        'code_length': 77,
        'comments': 0,
        'cyclomatic_complexity': 0,
        'indents': 0,
        'loop_count': 0,
        'line_length': 77,
        'identifiers': 0,
        'readability': 0
    },
    {
        'problem_title': 'CSS with No Formatting',
        'python_solutions': 'body{background-color:#f0f0f0;font-family:Arial,sans-serif}',
        'difficulty': 1,
        'num_of_lines': 1,
        'code_length': 59,
        'comments': 0,
        'cyclomatic_complexity': 0,
        'indents': 0,
        'loop_count': 0,
        'line_length': 59,
        'identifiers': 3,
        'readability': 0
    },
    {
        'problem_title': 'Python with Complex Logic',
        'python_solutions': '''def complex_function(a, b):\n    if a > b:\n        if a > 10:\n            if b < 5:\n                return a * b\n            else:\n                return a + b\n        else:\n            return b - a\n    else:\n        return a / b''',
        'difficulty': 3,
        'num_of_lines': 10,
        'code_length': 153,
        'comments': 0,
        'cyclomatic_complexity': 5,
        'indents': 4,
        'loop_count': 0,
        'line_length': 153,
        'identifiers': 6,
        'readability': 1
    }
]

# Convert the list of dictionaries to a DataFrame
new_data = pd.DataFrame(non_readable_snippets)

# Concatenate the new data to the existing DataFrame
df = pd.concat([df, new_data], ignore_index=True)

# Save the updated DataFrame back to CSV
df.to_csv('mnt/data/updated_code_snippets.csv', index=False)
