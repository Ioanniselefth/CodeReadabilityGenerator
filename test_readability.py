import unittest
import requests

class TestCodeReadability(unittest.TestCase):

    def setUp(self):
        self.base_url = 'http://127.0.0.1:5000/predict'

    def test_readable_code_snippets(self):
        readable_snippets = [
            """def calculate_area(radius):
                return 3.14 * radius * radius""",
            """for (let i = 0; i < 10; i++) {
                console.log(i);
            }""",
            """public class Calculator {
                public int add(int a, int b) {
                    return a + b;
                }
            }""",
            """<!DOCTYPE html>
            <html>
            <head>
                <title>Sample Form</title>
            </head>
            <body>
                <form>
                    <label for="name">Name:</label>
                    <input type="text" id="name" name="name"><br><br>
                    <input type="submit" value="Submit">
                </form>
            </body>
            </html>"""
        ]
        for snippet in readable_snippets:
            with self.subTest(snippet=snippet):
                response = requests.post(self.base_url, data={'snippet': snippet})
                self.assertIn('Readable', response.text)

    def test_non_readable_code_snippets(self):
        non_readable_snippets = [
            """def calculate_area(radius):
            return 3.14 * radius * radius""",
            """1234""",
            """SELECT name,age FROM users WHERE age>20AND city='New York';""",
            """function checkAge(age {
            if age < 18 {
            console.log("Minor");
            } else {
            console.log("Adult");
            }
            }""",
            """<html>
            <head><title>Sample Page</title></head>
            <body><h1>Title</h1><p>Content here
            </body>
            </html>""",
            """body{background-color:#f0f0f0;font-family:Arial,sans-serif}""",
            """def complex_function(a, b):
                if a > b:
                    if a > 10:
                        if b < 5:
                            return a * b
                        else:
                            return a + b
                    else:
                        return b - a
                else:
                    return a / b"""
        ]
        for snippet in non_readable_snippets:
            with self.subTest(snippet=snippet):
                response = requests.post(self.base_url, data={'snippet': snippet})
                self.assertIn('Not Readable', response.text)

if __name__ == '__main__':
    unittest.main()
