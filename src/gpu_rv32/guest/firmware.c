int mini_python(const char *, char *, unsigned, int *, unsigned);
char py_source[4096], py_output[4096];
int py_value, py_error;
unsigned py_budget;

int run_python(void) {
    py_error = mini_python(py_source, py_output, sizeof(py_output), &py_value, py_budget);
    return py_value;
}
