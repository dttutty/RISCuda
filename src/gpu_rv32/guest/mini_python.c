/*
 * Freestanding Python subset, executed as compiled RV32IM machine code.
 * Interface: mini_python(source, output, capacity, result, statement_budget).
 * Integers are signed 32-bit values; arithmetic wraps modulo 2^32. Literals
 * are decimal or hexadecimal, at most 0xffffffff. // and % follow Python's
 * floor rule. Negative shift counts fail; shifts >=32 saturate to zero/sign.
 * Statements: name = expression, expression, print(expression), pass,
 * indentation-based if/else and while. Expressions support parentheses,
 * + - * // %, & | ^ ~ << >>, comparisons, and/or/not, True and False.
 * There are no strings, containers, functions, imports, break or continue.
 * Indentation uses spaces (arbitrary consistent widths); tabs are rejected.
 * Blank lines, # comments and CRLF are accepted. No libc or allocation.
 * Limits: source4095 bytes,256 nonblank lines,32 variables,31-byte names,
 * 16 nested blocks and16 nested parentheses/unary operators.
 * Returns:0 success,1 syntax,2 unknown name,3 division by zero,4 capacity or
 * nesting limit,5 statement budget exhausted,6 negative shift count.
 * Output is NUL-terminated print lines. Result is the last assigned,
 * expression or printed value; conditions and pass do not replace it.
 */

enum { MP_LINES = 256, MP_VARIABLES = 32, MP_NAME = 32, MP_DEPTH = 16 };
enum { ST_EXPR, ST_ASSIGN, ST_PRINT, ST_IF, ST_ELSE, ST_WHILE, ST_PASS };

typedef struct {
    unsigned begin, end, indent, kind, target;
} Statement;

typedef struct {
    const char *source;
    char *output;
    unsigned output_capacity, output_length, line_count, variable_count;
    int error, last;
    Statement lines[MP_LINES];
    char names[MP_VARIABLES][MP_NAME];
    int values[MP_VARIABLES];
} Machine;

typedef struct {
    Machine *machine;
    const char *pos, *end;
    unsigned depth;
} Parser;

static int first_name(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}
static int rest_name(char c) { return first_name(c) || (c >= '0' && c <= '9'); }
static int equal(const char *a, const char *b) {
    while (*a && *a == *b) { ++a; ++b; }
    return *a == *b;
}
static int reserved(const char *name) {
    const char *keywords[] = {
        "False", "None", "True", "and", "as", "assert", "async", "await",
        "break", "class", "continue", "def", "del", "elif", "else", "except",
        "finally", "for", "from", "global", "if", "import", "in", "is", "lambda",
        "nonlocal", "not", "or", "pass", "raise", "return", "try", "while", "with", "yield"
    };
    for (unsigned i = 0; i < sizeof(keywords) / sizeof(keywords[0]); ++i)
        if (equal(name, keywords[i])) return 1;
    return 0;
}
static void fail(Machine *m, int error) { if (!m->error) m->error = error; }
static void spaces(Parser *p) { while (p->pos < p->end && *p->pos == ' ') ++p->pos; }

static int take(Parser *p, const char *token) {
    const char *at;
    spaces(p);
    at = p->pos;
    while (*token) {
        if (at == p->end || *at != *token) return 0;
        ++at; ++token;
    }
    p->pos = at;
    return 1;
}
static int word(Parser *p, const char *token) {
    const char *at;
    spaces(p);
    at = p->pos;
    while (*token) {
        if (at == p->end || *at != *token) return 0;
        ++at; ++token;
    }
    if (at < p->end && rest_name(*at)) return 0;
    p->pos = at;
    return 1;
}
static int identifier(Parser *p, char *name) {
    unsigned n = 0;
    spaces(p);
    if (p->pos == p->end || !first_name(*p->pos)) return 0;
    while (p->pos < p->end && rest_name(*p->pos)) {
        if (n == MP_NAME - 1) { fail(p->machine, 4); return 0; }
        name[n++] = *p->pos++;
    }
    name[n] = 0;
    return 1;
}
static int variable(Machine *m, const char *name, int create) {
    unsigned i, j;
    for (i = 0; i < m->variable_count; ++i)
        if (equal(name, m->names[i])) return (int)i;
    if (!create) { fail(m, 2); return 0; }
    if (i == MP_VARIABLES) { fail(m, 4); return 0; }
    for (j = 0; name[j]; ++j) m->names[i][j] = name[j];
    m->names[i][j] = 0;
    m->values[i] = 0;
    ++m->variable_count;
    return (int)i;
}

static int expression(Parser *p, int active);
static int primary(Parser *p, int active) {
    char name[MP_NAME];
    unsigned value = 0, base = 10, digits = 0;
    spaces(p);
    if (take(p, "(")) {
        int result;
        if (++p->depth > MP_DEPTH) { fail(p->machine, 4); return 0; }
        result = expression(p, active);
        --p->depth;
        if (!take(p, ")")) fail(p->machine, 1);
        return result;
    }
    if (p->pos < p->end && *p->pos >= '0' && *p->pos <= '9') {
        if (p->end - p->pos >= 2 && p->pos[0] == '0' &&
            (p->pos[1] == 'x' || p->pos[1] == 'X')) {
            base = 16; p->pos += 2;
        }
        while (p->pos < p->end) {
            char c = *p->pos;
            unsigned d = c >= '0' && c <= '9' ? (unsigned)(c - '0') :
                         c >= 'a' && c <= 'f' ? (unsigned)(c - 'a' + 10) :
                         c >= 'A' && c <= 'F' ? (unsigned)(c - 'A' + 10) : 16;
            if (d >= base) break;
            if (value > (0xffffffffu - d) / base) { fail(p->machine, 4); return 0; }
            value = value * base + d;
            ++digits; ++p->pos;
        }
        if (!digits) fail(p->machine, 1);
        /* Python rejects nonzero decimal literals with a leading zero. */
        if (base == 10 && digits > 1 && value && p->pos[-(int)digits] == '0') fail(p->machine, 1);
        return (int)value;
    }
    if (word(p, "True")) return 1;
    if (word(p, "False")) return 0;
    if (identifier(p, name)) {
        int slot;
        if (reserved(name)) { fail(p->machine, 1); return 0; }
        if (!active) return 0;
        slot = variable(p->machine, name, 0);
        return p->machine->error ? 0 : p->machine->values[slot];
    }
    fail(p->machine, 1);
    return 0;
}

static int unary(Parser *p, int active) {
    int kind = take(p, "+") ? 1 : take(p, "-") ? 2 : take(p, "~") ? 3 : 0;
    if (kind) {
        int value;
        if (++p->depth > MP_DEPTH) { fail(p->machine, 4); return 0; }
        value = unary(p, active);
        --p->depth;
        if (!active) return 0;
        return kind == 2 ? (int)(0u - (unsigned)value) : kind == 3 ? ~value : value;
    }
    return primary(p, active);
}

static int divide(Parser *p, int a, int b, int remainder) {
    int q, r;
    if (!b) { fail(p->machine, 3); return 0; }
    if (a == (int)0x80000000u && b == -1) return remainder ? 0 : a;
    q = a / b; r = a % b;
    if (r && ((a < 0) != (b < 0))) {
        q = (int)((unsigned)q - 1u);
        r = (int)((unsigned)r + (unsigned)b);
    }
    return remainder ? r : q;
}
static int product(Parser *p, int active) {
    int left = unary(p, active);
    while (!p->machine->error) {
        int op = take(p, "//") ? 1 : take(p, "*") ? 2 : take(p, "%") ? 3 : 0;
        int right;
        if (!op) break;
        right = unary(p, active);
        if (active && !p->machine->error)
            left = op == 2 ? (int)((unsigned)left * (unsigned)right) : divide(p, left, right, op == 3);
    }
    return left;
}
static int sum(Parser *p, int active) {
    int left = product(p, active);
    while (!p->machine->error) {
        int op = take(p, "+") ? 1 : take(p, "-") ? 2 : 0;
        int right;
        if (!op) break;
        right = product(p, active);
        if (active) left = op == 1 ? (int)((unsigned)left + (unsigned)right) :
                                    (int)((unsigned)left - (unsigned)right);
    }
    return left;
}
static int shift(Parser *p, int active) {
    int left = sum(p, active);
    while (!p->machine->error) {
        int op = take(p, "<<") ? 1 : take(p, ">>") ? 2 : 0;
        int right;
        if (!op) break;
        right = sum(p, active);
        if (!active || p->machine->error) continue;
        if (right < 0) { fail(p->machine, 6); return 0; }
        if (right >= 32) left = op == 1 ? 0 : left < 0 ? -1 : 0;
        else left = op == 1 ? (int)((unsigned)left << right) : left >> right;
    }
    return left;
}
static int bit_and(Parser *p, int active) {
    int left = shift(p, active);
    while (!p->machine->error && take(p, "&")) {
        int right = shift(p, active);
        if (active) left &= right;
    }
    return left;
}
static int bit_xor(Parser *p, int active) {
    int left = bit_and(p, active);
    while (!p->machine->error && take(p, "^")) {
        int right = bit_and(p, active);
        if (active) left ^= right;
    }
    return left;
}
static int bit_or(Parser *p, int active) {
    int left = bit_xor(p, active);
    while (!p->machine->error && take(p, "|")) {
        int right = bit_xor(p, active);
        if (active) left |= right;
    }
    return left;
}
static int comparison(Parser *p, int active) {
    int left = bit_or(p, active), result = 1, seen = 0;
    while (!p->machine->error) {
        int op = take(p, "==") ? 1 : take(p, "!=") ? 2 : take(p, "<=") ? 3 :
                 take(p, ">=") ? 4 : take(p, "<") ? 5 : take(p, ">") ? 6 : 0;
        int right, yes;
        if (!op) break;
        right = bit_or(p, active && result);
        yes = op == 1 ? left == right : op == 2 ? left != right : op == 3 ? left <= right :
              op == 4 ? left >= right : op == 5 ? left < right : left > right;
        if (active) result = result && yes;
        left = right; seen = 1;
    }
    return seen ? result : left;
}
static int logical_not(Parser *p, int active) {
    if (word(p, "not")) {
        int value;
        if (++p->depth > MP_DEPTH) { fail(p->machine, 4); return 0; }
        value = logical_not(p, active);
        --p->depth;
        return active ? !value : 0;
    }
    return comparison(p, active);
}
static int logical_and(Parser *p, int active) {
    int left = logical_not(p, active);
    while (!p->machine->error && word(p, "and")) {
        int right = logical_not(p, active && left != 0);
        if (active && left) left = right;
    }
    return left;
}
static int expression(Parser *p, int active) {
    int left = logical_and(p, active);
    while (!p->machine->error && word(p, "or")) {
        int right = logical_and(p, active && left == 0);
        if (active && !left) left = right;
    }
    return left;
}

static void print_value(Machine *m, int value) {
    char digits[10];
    unsigned n = 0, magnitude = value < 0 ? 0u - (unsigned)value : (unsigned)value;
    unsigned required;
    do { digits[n++] = (char)('0' + magnitude % 10); magnitude /= 10; } while (magnitude);
    required = n + (value < 0 ? 1u : 0u) + 1u;
    if (required >= m->output_capacity - m->output_length) { fail(m, 4); return; }
    if (value < 0) m->output[m->output_length++] = '-';
    while (n) m->output[m->output_length++] = digits[--n];
    m->output[m->output_length++] = '\n';
    m->output[m->output_length] = 0;
}

static Parser parser_for(Machine *m, Statement *s) {
    Parser p;
    p.machine = m; p.pos = m->source + s->begin; p.end = m->source + s->end; p.depth = 0;
    return p;
}
static void finished(Parser *p) {
    spaces(p);
    if (p->pos != p->end) fail(p->machine, 1);
}

/* Parse once without evaluation to reject bad syntax even in skipped blocks. */
static void classify(Machine *m, Statement *s) {
    Parser p = parser_for(m, s);
    char name[MP_NAME];
    if (word(&p, "if")) s->kind = ST_IF;
    else if (word(&p, "while")) s->kind = ST_WHILE;
    else if (word(&p, "else")) s->kind = ST_ELSE;
    else if (word(&p, "print")) s->kind = ST_PRINT;
    else if (word(&p, "pass")) s->kind = ST_PASS;
    else {
        const char *save = p.pos;
        s->kind = ST_EXPR;
        if (identifier(&p, name)) {
            spaces(&p);
            if (p.pos < p.end && p.pos[0] == '=' && (p.pos + 1 == p.end || p.pos[1] != '=')) {
                if (reserved(name)) { fail(m, 1); return; }
                ++p.pos; s->kind = ST_ASSIGN;
            }
        }
        if (s->kind == ST_EXPR) p.pos = save;
    }
    if (s->kind == ST_PRINT && !take(&p, "(")) { fail(m, 1); return; }
    if (s->kind != ST_ELSE && s->kind != ST_PASS) (void)expression(&p, 0);
    if (s->kind == ST_PRINT && !take(&p, ")")) fail(m, 1);
    if ((s->kind == ST_IF || s->kind == ST_WHILE || s->kind == ST_ELSE) && !take(&p, ":")) fail(m, 1);
    finished(&p);
}

static void scan(Machine *m) {
    unsigned length = 0, pos = 0, i;
    while (m->source[length]) {
        if (length == 4095) { fail(m, 4); return; }
        ++length;
    }
    while (pos < length && !m->error) {
        unsigned begin = pos, indent = 0, end;
        while (pos < length && m->source[pos] != '\n') ++pos;
        end = pos;
        if (pos < length) ++pos;
        if (end > begin && m->source[end - 1] == '\r') --end;
        while (begin < end && m->source[begin] == ' ') { ++begin; ++indent; }
        for (i = begin; i < end; ++i) {
            if (m->source[i] == '#') { end = i; break; }
            if (m->source[i] == '\t' || m->source[i] == '\r') { fail(m, 1); return; }
        }
        while (end > begin && m->source[end - 1] == ' ') --end;
        if (begin == end) continue;
        if (m->line_count == MP_LINES) { fail(m, 4); return; }
        m->lines[m->line_count].begin = begin;
        m->lines[m->line_count].end = end;
        m->lines[m->line_count].indent = indent;
        m->lines[m->line_count].target = 0;
        classify(m, &m->lines[m->line_count]);
        ++m->line_count;
    }
}

static void compile_block(Machine *m, unsigned *at, unsigned indent, unsigned depth) {
    if (depth > MP_DEPTH) { fail(m, 4); return; }
    while (*at < m->line_count && !m->error) {
        unsigned header = *at;
        Statement *s = &m->lines[header];
        if (s->indent < indent) return;
        if (s->indent != indent || s->kind == ST_ELSE) { fail(m, 1); return; }
        ++*at;
        if (s->kind != ST_IF && s->kind != ST_WHILE) continue;
        if (*at == m->line_count || m->lines[*at].indent <= indent) { fail(m, 1); return; }
        compile_block(m, at, m->lines[*at].indent, depth + 1);
        if (m->error) return;
        if (s->kind == ST_IF && *at < m->line_count &&
            m->lines[*at].indent == indent && m->lines[*at].kind == ST_ELSE) {
            unsigned alternative = *at;
            ++*at;
            s->target = *at;
            if (*at == m->line_count || m->lines[*at].indent <= indent) { fail(m, 1); return; }
            compile_block(m, at, m->lines[*at].indent, depth + 1);
            m->lines[alternative].target = *at;
        } else s->target = *at;
    }
}

static int evaluate(Machine *m, Statement *s) {
    Parser p = parser_for(m, s);
    char name[MP_NAME];
    int value;
    if (s->kind == ST_ASSIGN) {
        int slot;
        if (!identifier(&p, name) || !take(&p, "=")) { fail(m, 1); return 0; }
        value = expression(&p, 1);
        if (m->error) return 0;
        slot = variable(m, name, 1);
        if (!m->error) m->values[slot] = value;
    } else {
        if (s->kind == ST_IF) (void)word(&p, "if");
        else if (s->kind == ST_WHILE) (void)word(&p, "while");
        else if (s->kind == ST_PRINT) { (void)word(&p, "print"); (void)take(&p, "("); }
        value = expression(&p, 1);
    }
    if (!m->error && s->kind != ST_IF && s->kind != ST_WHILE) {
        if (s->kind == ST_PRINT) print_value(m, value);
        if (!m->error) m->last = value;
    }
    return value;
}

int mini_python(const char *source, char *output, unsigned output_capacity,
                int *result, unsigned statement_budget) {
    Machine m;
    unsigned at = 0, pc = 0, loop_count = 0;
    unsigned loop_header[MP_DEPTH], loop_end[MP_DEPTH];
    if (!source || !output || !result || !output_capacity) return 4;
    m.source = source; m.output = output; m.output_capacity = output_capacity;
    m.output_length = m.line_count = m.variable_count = 0;
    m.error = m.last = 0;
    output[0] = 0; *result = 0;
    scan(&m);
    if (!m.error) compile_block(&m, &at, 0, 0);
    if (!m.error && at != m.line_count) fail(&m, 1);
    while (!m.error) {
        Statement *s;
        int value;
        if (loop_count && pc == loop_end[loop_count - 1]) pc = loop_header[--loop_count];
        if (pc == m.line_count) break;
        if (!statement_budget) { fail(&m, 5); break; }
        --statement_budget;
        s = &m.lines[pc];
        if (s->kind == ST_ELSE) { pc = s->target; continue; }
        if (s->kind == ST_PASS) { ++pc; continue; }
        value = evaluate(&m, s);
        if (m.error) break;
        if (s->kind == ST_IF || s->kind == ST_WHILE) {
            if (!value) { pc = s->target; continue; }
            if (s->kind == ST_WHILE) {
                if (loop_count == MP_DEPTH) { fail(&m, 4); break; }
                loop_header[loop_count] = pc;
                loop_end[loop_count++] = s->target;
            }
        }
        ++pc;
    }
    *result = m.last;
    return m.error;
}
