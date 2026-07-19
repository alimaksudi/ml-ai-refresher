"""Builder for Lesson PRE-02 — Algebra, Functions, and Graphs."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nbbuild import build, code, md  # noqa: E402


cells = [
    md(r"""
    # PRE-02 · Algebra, Functions, and Graphs

    *How a changing quantity becomes a rule, a table, and a picture*

    | Lesson detail | Value |
    | --- | --- |
    | Prerequisite | PRE-01, or equivalent arithmetic and symbol fluency |
    | Estimated study time | 3–4 hours, including practice |
    | Main outcome | Explain and inspect a simple relationship in words, algebra, a table, and a graph |
    | Next lesson | PRE-03 · Python, NumPy, and Jupyter Foundations |

    You do not need to be a “math person” to learn algebra. Algebra is ordinary
    reasoning written in a compact way. We will unpack every compact line before
    using it.
    """),

    md(r"""
    ## 1 · What you will be able to do

    By the end of this lesson, you will be able to:

    - tell a variable, constant, coefficient, expression, and equation apart;
    - solve a one-variable equation without using a sign-changing shortcut;
    - solve a simple inequality and explain why a negative multiplier reverses it;
    - read and evaluate function notation such as $f(3)$;
    - describe domain and range in plain language;
    - move between a real situation, formula, value table, coordinates, and graph;
    - explain slope and intercept with units;
    - recognize when a straight-line rule is useful and when it is misleading.

    The goal is not to manipulate symbols quickly. The goal is to know what the
    symbols claim about a real situation.
    """),

    md(r"""
    ## 2 · The problem we are trying to solve

    A delivery service charges a fixed booking fee of **$5**, then **$2 for every
    kilometre** travelled.

    We could calculate each trip separately:

    - 0 km costs $5;
    - 1 km costs $7;
    - 2 km costs $9;
    - 3 km costs $11.

    That works for four trips. It becomes annoying for hundreds of distances.
    More importantly, a list does not make the relationship easy to inspect.

    We need one reusable rule that answers all of these questions:

    - What does any distance cost?
    - How quickly does cost rise?
    - Is there a cost even when distance is zero?
    - Can we trust the same rule at 1,000 km?

    Algebra stores the rule. A function names it. A graph lets us see its shape.
    """),

    md(r"""
    ## 3 · Three mental pictures

    ### The balance scale

    An equation says that two amounts are equal. Imagine one amount on each side
    of a balanced scale. If you remove 3 from the left side, remove 3 from the
    right side as well. This is why solving equations works.

    The analogy has a limit: an equation is a statement about equal values, not a
    physical object. The useful part is the idea of preserving balance.

    ### The function machine

    <div style="display: flex; align-items: center; justify-content: center; gap: 14px; margin: 24px 0; flex-wrap: wrap;">
      <div style="border: 2px solid #4c78a8; border-radius: 10px; padding: 14px 20px; text-align: center; background: #eef5ff; color: #172b4d;">
        <strong>Input</strong><br>
        distance $d$
      </div>
      <div style="font-size: 26px; color: #555;" aria-label="then">→</div>
      <div style="border: 2px solid #f28e2b; border-radius: 10px; padding: 14px 20px; text-align: center; background: #fff4e8; color: #4a2b0b;">
        <strong>Rule</strong><br>
        multiply by $2$, then add $5$
      </div>
      <div style="font-size: 26px; color: #555;" aria-label="produces">→</div>
      <div style="border: 2px solid #59a14f; border-radius: 10px; padding: 14px 20px; text-align: center; background: #eef8ec; color: #173d17;">
        <strong>Output</strong><br>
        cost $C(d)$
      </div>
    </div>

    The machine picture helps us follow one input through one rule. A mathematical
    function is more precise than a machine: every allowed input must have exactly
    one output.

    ### The graph as a map

    A graph is a map of input-output pairs. Moving right changes the input. Moving
    up or down shows how the output changes. One point is one case; the whole line
    shows the rule across many cases.
    """),

    md(r"""
    ## 4 · Build the mathematical language

    ### 4.1 Variables, constants, coefficients, and terms

    Start with the delivery rule:

    $$
    C = 2d + 5
    $$

    Read it aloud: “Cost equals two times distance plus five.”

    - $C$ is the total cost. It is a **variable** because it can change.
    - $d$ is distance in kilometres. It is also a variable.
    - $2$ is a **coefficient**: a number multiplying a variable.
    - $5$ is a **constant**: it stays fixed in this rule.
    - $2d$ and $5$ are **terms**: pieces joined by addition or subtraction.
    - $2d + 5$ is an **expression**: a calculation without an equals sign.
    - $C = 2d + 5$ is an **equation**: it says two quantities have equal value.

    In algebra, writing $2d$ means $2\times d$. The multiplication sign is omitted
    to keep formulas compact.
    """),

    md(r"""
    ### 4.2 Evaluate an expression by substitution

    Suppose the trip is 3 km. **Substitution** means replacing a symbol with its
    known value.

    **Symbols:** $C$ means cost in dollars, $d$ means distance in kilometres, and
    the equals sign says the expressions on both sides have the same value.

    $$
    C = 2d + 5
    $$

    Replace $d$ with $3$:

    $$
    C = 2(3) + 5
    $$

    Multiply before adding:

    $$
    C = 6 + 5 = 11
    $$

    The result is **$11**, not just 11. The unit gives the answer meaning.

    **Self-check:** substitute $d=0$. You should get $C=5$, matching the fixed
    booking fee.
    """),

    md(r"""
    ### 4.3 Solve an equation by undoing operations

    Evaluation asks, “What is the output for this input?” Solving reverses the
    question: “Which input produced this output?”

    A customer paid $17. What distance did they travel?

    **Symbols:** $d$ is the unknown distance, $2$ is the dollars-per-kilometre
    coefficient, $5$ is the fixed fee, and $17$ is the known total cost.

    $$
    2d + 5 = 17
    $$

    The operations applied to $d$ were:

    1. multiply by 2;
    2. add 5.

    Undo them in reverse order. First subtract 5 from **both** sides:

    $$
    2d + 5 - 5 = 17 - 5
    $$

    Therefore:

    $$
    2d = 12
    $$

    Divide **both** sides by 2:

    $$
    \frac{2d}{2} = \frac{12}{2}
    $$

    Therefore:

    $$
    d = 6
    $$

    Check by substitution: $2(6)+5=17$. The solution is 6 km.

    Avoid the shortcut “move 5 across and change its sign” until the balance idea
    is automatic. That shortcut hides the reason and causes mistakes later.
    """),

    md(r"""
    ### 4.4 Equations with negatives and fractions

    The same balance rule still works when a coefficient is negative.

    **Symbols:** $x$ is the unknown number. The minus sign makes its coefficient
    negative, and the fraction bar later in this section means division.

    $$
    -3x + 4 = 10
    $$

    Subtract 4 from both sides:

    $$
    -3x = 6
    $$

    Divide both sides by $-3$:

    $$
    x = -2
    $$

    Check: $-3(-2)+4=6+4=10$.

    A fraction is also a division instruction. For

    $$
    \frac{x}{4}=3
    $$

    multiply both sides by 4:

    $$
    x=12
    $$

    You do not need a different trick for each equation. Ask which operation was
    applied to the unknown, then apply its inverse to both sides.
    """),

    md(r"""
    ### 4.5 Inequalities describe a region, not one value

    An equation such as $x=4$ identifies one value. An inequality identifies many
    values.

    **Symbols:** $x$ is the value being constrained. The signs $<$ and $>$ mean
    “less than” and “greater than,” so they describe which side of a boundary is
    allowed.

    $$
    2x + 1 < 9
    $$

    Subtract 1 from both sides, then divide by 2:

    $$
    2x < 8
    $$

    $$
    x < 4
    $$

    Every number below 4 works. For example, $x=3$ gives $7<9$.

    There is one important rule: multiplying or dividing both sides by a negative
    number reverses the inequality.

    $$
    -2x < 6
    $$

    Divide by $-2$:

    $$
    x > -3
    $$

    Why reverse it? Multiplying by $-1$ mirrors numbers across zero. Although
    $2<5$, their negatives satisfy $-2>-5$. The order flips on the number line.

    Check $x=0$: $-2(0)=0<6$, so $0>-3$ belongs to the solution region.
    """),

    md(r"""
    ### 4.6 A function names an input-output rule

    We can name the delivery rule $C$:

    **Symbols:** $C$ names the cost function, $d$ is its distance input, and
    $C(d)$ means the output produced for distance $d$.

    $$
    C(d)=2d+5
    $$

    Read it aloud: “C of d equals two d plus five.”

    - $C$ is the function name.
    - $d$ inside parentheses is the input.
    - $2d+5$ is the rule.
    - $C(3)$ means “the output of function $C$ when the input is 3.”

    Evaluate it:

    $$
    C(3)=2(3)+5=11
    $$

    Parentheses in $C(3)$ do **not** mean multiplication. They show the function's
    input.

    A function may give the same output for different inputs. For example,
    $f(x)=x^2$ gives $f(2)=4$ and $f(-2)=4$. But one allowed input cannot have two
    different outputs under the same deterministic rule.
    """),

    md(r"""
    ### 4.7 Domain and range set the boundaries

    The **domain** is the set of allowed inputs. The **range** is the set of outputs
    the function can produce for that domain.

    **Symbols:** $d$ is distance, $C$ is cost, and $\ge$ means “greater than or
    equal to,” including the boundary value.

    For a real delivery distance, negative kilometres do not make sense. A useful
    domain is:

    $$
    d \ge 0
    $$

    Because $C(d)=2d+5$, the smallest cost occurs at $d=0$. Therefore the range is:

    $$
    C \ge 5
    $$

    Algebra alone may allow negative inputs, but the real situation can forbid
    them. Domain is part of the model, not a decorative note.

    A second example: $g(x)=1/x$ cannot use $x=0$ because division by zero is
    undefined. Here the formula itself removes zero from the domain.
    """),

    md(r"""
    ### 4.8 Coordinates and tables connect rules to graphs

    A coordinate is written $(x,y)$:

    **Symbols:** $x$ names the horizontal coordinate and $y$ names the vertical
    coordinate. Parentheses keep the ordered pair together.

    - $x$ gives the horizontal position;
    - $y$ gives the vertical position;
    - order matters, so $(2,9)$ differs from $(9,2)$.

    For the delivery rule, use distance as the horizontal coordinate and cost as
    the vertical coordinate.

    | Distance $d$ (km) | Cost $C(d)$ ($) | Coordinate $(d,C)$ |
    | ---: | ---: | ---: |
    | 0 | 5 | $(0,5)$ |
    | 1 | 7 | $(1,7)$ |
    | 2 | 9 | $(2,9)$ |
    | 3 | 11 | $(3,11)$ |

    Each row is one input-output pair. Plotting all coordinates reveals a straight
    line because the cost rises by the same amount each time distance rises by one.

    ```text
    cost ($)
      11 |             ●
       9 |         ●
       7 |     ●
       5 | ●
         +-----------------> distance (km)
           0   1   2   3
    ```

    Read the picture from the axes before reading the dots. The horizontal spacing
    represents kilometres. The vertical spacing represents dollars. The equal rise
    between neighbouring dots is the visual clue that the rate is constant.
    """),

    md(r"""
    ### 4.9 Slope and intercept explain a straight line

    A linear function is often written as:

    **Symbols:** $x$ is input, $y$ is output, $m$ is slope, and $b$ is the vertical
    intercept. Subscripts later label which point a coordinate belongs to.

    $$
    y=mx+b
    $$

    - $x$ is the input.
    - $y$ is the output.
    - $m$ is the **slope**, the output change per one unit of input change.
    - $b$ is the **intercept**, the output when $x=0$.

    Slope between two points is:

    $$
    m=\frac{y_2-y_1}{x_2-x_1}
    $$

    The subscripts 1 and 2 label the first and second point. The numerator is the
    change in output. The denominator is the change in input.

    Using $(1,7)$ and $(3,11)$:

    $$
    m=\frac{11-7}{3-1}=\frac{4}{2}=2
    $$

    The slope is **$2 per kilometre**. Its units are dollars divided by kilometres.
    The intercept is $5 because $C(0)=5$.

    - Positive slope: the line rises from left to right.
    - Negative slope: the line falls.
    - Zero slope: the output stays constant.
    - Vertical line: slope is undefined because the input change is zero.
    """),

    md(r"""
    ### 4.10 Linear does not mean “always correct”

    A straight line has a constant rate of change. The delivery rule adds exactly
    $2 for every kilometre, so it is linear.

    **Symbols:** in $y=x^2$, $x$ is the input, $y$ is the output, and the exponent
    $2$ means multiply $x$ by itself.

    A curved rule changes at a changing rate. For example:

    $$
    y=x^2
    $$

    From $x=1$ to $x=2$, the output rises from 1 to 4. From $x=2$ to $x=3$, it
    rises from 4 to 9. The increase is not constant, so one straight line cannot
    describe the whole curve.

    A linear rule may still be a useful approximation over a small range. It
    becomes risky when we **extrapolate**, meaning predict far beyond the observed
    range. A city delivery price rule should not automatically be trusted for a
    1,000 km trip.
    """),

    md(r"""
    ## 5 · Worked example: from words to five representations

    A taxi charges a $4 starting fee and $1.50 per kilometre.

    **Symbols:** $F$ names the fare function and $d$ is distance in kilometres.
    The value $F(d)$ is the fare in dollars for that distance.

    **Words:** start at $4, then add $1.50 for every kilometre.

    **Formula:**

    $$
    F(d)=1.5d+4
    $$

    Here $F$ is fare in dollars and $d$ is distance in kilometres.

    **Manual calculation at 6 km:**

    $$
    F(6)=1.5(6)+4=9+4=13
    $$

    **Table:**

    | $d$ (km) | $F(d)$ ($) |
    | ---: | ---: |
    | 0 | 4.00 |
    | 2 | 7.00 |
    | 4 | 10.00 |
    | 6 | 13.00 |

    **Coordinates:** $(0,4)$, $(2,7)$, $(4,10)$, and $(6,13)$.

    **Graph meaning:** the line starts at 4 and rises $1.50 for each kilometre.

    **Check:** the fare at 6 km must exceed both the $4 starting fee and the 4 km
    fare. It does.
    """),

    md(r"""
    ## 6 · A small Python preview

    PRE-03 teaches Python carefully. For now, read this as the same algebra written
    for a computer. Predict each result before running the cell.

    The code deliberately repeats the formula. That repetition will create a good
    reason to learn Python functions and loops in the next lesson.
    """),

    code(r"""
    starting_fee_dollars = 4.0
    price_per_km = 1.5

    distance_0_km = 0
    fare_0_dollars = price_per_km * distance_0_km + starting_fee_dollars
    print("0 km fare:", fare_0_dollars)

    distance_2_km = 2
    fare_2_dollars = price_per_km * distance_2_km + starting_fee_dollars
    print("2 km fare:", fare_2_dollars)

    distance_6_km = 6
    fare_6_dollars = price_per_km * distance_6_km + starting_fee_dollars
    print("6 km fare:", fare_6_dollars)

    assert fare_0_dollars == 4.0
    assert fare_2_dollars == 7.0
    assert fare_6_dollars == 13.0
    """),

    md(r"""
    The expected outputs are `4.0`, `7.0`, and `13.0`. Each number matches the
    manual table. The assertions are executable checks: they stay quiet when the
    claim is true and stop with an error when it is false.

    Notice what is missing: there is no reusable Python function yet. PRE-03 will
    turn the repeated calculation into code that accepts many inputs safely.
    """),

    md(r"""
    ## 7 · Read a graph before trusting it

    A useful graph needs:

    - a labelled horizontal axis with units;
    - a labelled vertical axis with units;
    - a sensible scale;
    - points or a line that match the stated rule;
    - the observed input range, so extrapolation is visible.

    Before interpreting a line, ask:

    1. What does each axis measure?
    2. What does one point mean?
    3. Is the line showing observations, a mathematical rule, or a model's
       predictions?
    4. Does the axis start at zero? If not, does the scale exaggerate a difference?
    5. Are we looking inside or outside the observed range?

    A graph can show association, but it cannot prove that one variable causes the
    other. Ice-cream sales and sunburn cases may rise together because hot weather
    affects both.
    """),

    md(r"""
    ## 8 · When to use each representation

    Use algebra and functions when:

    - one repeatable rule connects input and output;
    - you need to calculate many cases;
    - you want to expose parameters such as a rate and starting value.

    Use a table when:

    - there are only a few exact cases;
    - the rule is unknown;
    - you want readers to look up values directly.

    Use a graph when:

    - shape, trend, turning points, or unusual values matter;
    - you want to compare several relationships;
    - a visual pattern is easier to understand than a long table.

    Do not force a straight line onto a curved or changing relationship. Use a
    nonlinear rule, piecewise rule, or lookup table when those describe reality
    more honestly.
    """),

    md(r"""
    ## 9 · Real-world connection to machine learning

    A machine-learning model is also a function: features go in and a prediction
    comes out.

    A simple house-price model might say:

    **Symbols:** $s$ means size in square metres, $\widehat{P}$ means predicted
    price, and the hat marks an estimate rather than a known true value.

    $$
    \widehat{P}(s)=120s+50{,}000
    $$

    - $s$ is house size in square metres.
    - $\widehat{P}$, read “P hat,” is predicted price, not guaranteed true price.
    - $120$ is the predicted price increase per additional square metre.
    - $50{,}000$ is the model's predicted price at zero square metres.

    That intercept may be mathematically necessary while having little practical
    meaning, because a zero-square-metre house is outside the useful domain.

    Later, linear regression will learn the coefficient and intercept from data.
    This lesson gives you the language needed to understand what it learns.
    """),

    md(r"""
    ## 10 · Common mistakes and how to catch them

    | Common mistake | Why it happens | Quick check |
    | --- | --- | --- |
    | Treating $2x$ as $2+x$ | The multiplication sign is hidden | Replace $x$ with 3: $2x$ must become 6 |
    | Changing a sign while “moving” a term | A shortcut replaced the balance idea | Write the operation on both sides |
    | Forgetting to reverse an inequality | Division by a negative feels routine | Test one value in the final region |
    | Reading $f(3)$ as $f\times3$ | Function parentheses look like grouping | Say “the output of $f$ at input 3” |
    | Swapping coordinates | Both values sit inside parentheses | Say horizontal first, vertical second |
    | Calling the coefficient the intercept | Both are numbers in the formula | Set $x=0$; the remaining output is the intercept |
    | Dropping units from slope | The calculation looks unitless | Write output units divided by input units |
    | Assuming every trend is linear | Straight lines are familiar | Compare changes across several equal input steps |
    | Treating association as causation | The graph looks persuasive | Ask what third variable could affect both |
    | Trusting distant extrapolation | The formula still returns a number | Mark the observed input range before predicting |
    """),

    md(r"""
    ## 11 · Compare the related ideas

    | Concept | Main purpose | Strength | Limitation | Best use |
    | --- | --- | --- | --- | --- |
    | Expression | Describe a calculation | Compact | Makes no equality claim | Computing a quantity |
    | Equation | State two values are equal | Can solve for unknowns | May have zero, one, or many solutions | Finding an unknown |
    | Inequality | Describe an allowed region | Represents limits | Needs careful sign handling | Thresholds and constraints |
    | Function | Name an input-output rule | Reusable | Domain must be clear | Repeated prediction or transformation |
    | Table | Store selected exact cases | Easy lookup | Does not show every possible input | Small finite data |
    | Graph | Show shape visually | Patterns are visible | Reading values may be approximate | Trend and comparison |
    | Linear rule | Model constant change | Simple and interpretable | Cannot capture changing rates globally | Stable rates or local approximations |
    | Nonlinear rule | Model changing rates | More flexible | Often harder to interpret | Curves and complex relationships |
    """),

    md(r"""
    ## 12 · Readiness check

    Try these without looking back.

    1. In $y=3x+8$, identify the input, output, coefficient, and constant.
    2. Evaluate $f(x)=4x-1$ at $x=3$.
    3. Solve $5x+2=22$ and check the answer by substitution.
    4. Solve $-2x<8$ and explain why the inequality reverses.
    5. Explain the slope and intercept of $C(d)=2d+5$ with units.
    6. State a realistic domain for delivery distance.
    7. Explain why a graph alone cannot prove causation.

    **Readiness threshold:** answer at least 6 of 7 correctly, including Questions
    3, 4, and 5. If those three are weak, repeat the balance, inequality, and slope
    sections using new numbers.
    """),

    md(r"""
    ## 13 · Teach it back

    Imagine a friend says, “A formula is just a bunch of letters.” Use the taxi
    example to explain:

    - what each symbol represents;
    - how words become $F(d)=1.5d+4$;
    - how one input becomes one output;
    - how the same rule becomes a table and coordinates;
    - what the slope and intercept mean;
    - why the rule should not be trusted at every possible distance.

    If you can explain those ideas without saying “it just moves to the other side,”
    your understanding is becoming durable.
    """),

    md(r"""
    ## 14 · Practice, self-check, and solutions

    **Estimated practice time:** 50–75 minutes.

    ### Worked example

    A streaming service charges $6 per month plus $2 for each rented film.

    Let $f$ be the number of films and $T$ be total monthly cost.

    $$
    T(f)=2f+6
    $$

    At 4 films:

    $$
    T(4)=2(4)+6=14
    $$

    The slope is $2 per film. The intercept is $6 per month when no films are
    rented.

    ### Guided practice

    1. Label the variables, coefficient, and constant in $y=5x-3$.
    2. Evaluate $g(x)=x^2+2$ at $x=-3$.
    3. Solve $3x+4=19$, showing the same operation on both sides.
    4. Solve $-4x\le12$, then test one value from your solution region.

    ### Independent practice

    5. A gym charges a $20 sign-up fee and $15 per month. Write a function for
       total cost after $m$ months. Calculate the cost after 6 months.
    6. Make a four-row value table for your gym function using $m=0,1,3,6$.
    7. Convert the table rows into coordinates. State the slope, intercept, and
       their units.
    8. Give one reason the gym rule might fail after several years.

    ### Challenge

    Plan A costs $10 plus $3 per gigabyte. Plan B costs $22 plus $1 per gigabyte.

    - Write one function for each plan.
    - Find the usage where the costs are equal.
    - Explain which plan is cheaper below and above that usage.
    - Check your conclusion with one value on each side of the meeting point.

    ### Self-check before reading solutions

    For every answer, ask:

    - Did I preserve equality by acting on both sides?
    - Did I reverse an inequality only after multiplying or dividing by a negative?
    - Did I keep units?
    - Does substitution confirm the result?
    - Does the answer make sense in the real situation?
    """),

    md(r"""
    ### Solution and scoring rubric

    **Guided practice**

    1. $x$ is the input, $y$ is the output, $5$ is the coefficient, and $-3$ is
       the constant.
    2. $g(-3)=(-3)^2+2=9+2=11$.
    3. Subtract 4 from both sides: $3x=15$. Divide both sides by 3: $x=5$.
       Check: $3(5)+4=19$.
    4. Divide by $-4$ and reverse the sign: $x\ge-3$. Test $x=0$:
       $-4(0)=0\le12$.

    **Independent practice**

    5. $G(m)=15m+20$. At 6 months, $G(6)=90+20=110$ dollars.
    6. The costs are $20$, $35$, $65$, and $110$ for months $0$, $1$, $3$, and
       $6$.
    7. Coordinates: $(0,20)$, $(1,35)$, $(3,65)$, and $(6,110)$. Slope: $15 per
       month. Intercept: $20 at zero months.
    8. The gym could change its fee, add annual charges, or offer a discount.

    **Challenge**

    **Symbols:** $A(g)$ and $B(g)$ are the two plan-cost functions, and $g$ is
    data usage in gigabytes. Setting the functions equal asks where their costs
    match.

    $$
    A(g)=3g+10
    $$

    $$
    B(g)=g+22
    $$

    Set them equal:

    $$
    3g+10=g+22
    $$

    Subtract $g$ and 10 from both sides:

    $$
    2g=12
    $$

    Therefore $g=6$. Both plans cost $28 at 6 GB. At 4 GB, A costs $22 and B
    costs $26, so A is cheaper. At 8 GB, A costs $34 and B costs $30, so B is
    cheaper.

    Award one point for each guided and independent answer, with Question 7 worth
    two points. Award four challenge points: two correct functions, correct meeting
    point, correct comparison, and two valid checks. Maximum: 13 points.

    **Common mistakes:** losing the negative sign in Question 1, squaring $-3$
    without parentheses, reversing an inequality during addition, omitting the
    fixed fee, or comparing plans without testing both sides of the meeting point.

    **Readiness threshold:** 10/13, including a correct equation solution,
    inequality solution, slope interpretation, and challenge meeting point.
    """),

    md(r"""
    ## Ready to move on?

    ### Quick check

    You are ready for PRE-03 when you can:

    - translate between words, a formula, a small table, and coordinates;
    - solve a two-step equation and verify it by substitution;
    - explain the negative-number inequality rule using the number line;
    - evaluate a function and describe its realistic domain;
    - explain slope and intercept with units;
    - say why a straight-line rule may fail outside its useful range.

    ### Teach it back

    Explain the delivery rule $C(d)=2d+5$ from the customer's story through its
    equation, table, graph shape, slope, intercept, domain, and limitation.

    ### Memory aid

    **A function is one rule connecting allowed inputs to outputs; a graph makes
    that connection visible.**

    PRE-03 comes next. It will turn these hand calculations into reusable Python
    without assuming you already know how to program.
    """),
]


build("00_prerequisites/02_algebra_functions_and_graphs.ipynb", cells)
