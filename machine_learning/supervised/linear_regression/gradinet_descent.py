#
# Gradient Descent For Simple Linear Regression
# ============================================================
#
# What this method does:
# - starts with guesses for the slope and intercept
# - measures how wrong the line is
# - nudges the parameters step by step toward lower error
#
# Why we use it:
# - it shows how models can learn numerically instead of by direct formulas
# - it is the foundation behind many larger machine learning algorithms
# - it makes the idea of optimization concrete

def get_gradient_at_b(x, y, m, b):
    diff = 0

    for i_x, i_y in zip(x, y):
        diff += i_y - (m * i_x + b)

    b_gradient = -2 / len(y) * diff

    return b_gradient


def get_gradient_at_m(x, y, m, b):
    diff = 0

    for i_x, i_y in zip(x, y):
        diff += i_x * (i_y - (m * i_x + b))

    m_gradient = -2 / len(y) * diff

    return m_gradient


def step_gradient(x, y, b_current, m_current):

    # IMPORTANT: m first, b second
    b_gradient = get_gradient_at_b(x, y, m_current, b_current)

    m_gradient = get_gradient_at_m(x, y, m_current, b_current)

    learning_rate = 0.01

    b = b_current - (learning_rate * b_gradient)

    m = m_current - (learning_rate * m_gradient)

    return b, m


def get_loss(x, y, m, b):

    total_error = 0

    for i_x, i_y in zip(x, y):

        prediction = m * i_x + b

        total_error += (i_y - prediction) ** 2

    return total_error / len(y)


def gradient_descent(x, y, b, m):

    previous_loss = float("inf")

    for i in range(1000):

        b, m = step_gradient(x, y, b, m)

        current_loss = get_loss(x, y, m, b)

        print(
            f"step {i:3d} | "
            f"b={b:.3f} | "
            f"m={m:.3f} | "
            f"loss={current_loss:.3f}"
        )

        # Stop when improvement is tiny
        if abs(previous_loss - current_loss) < 0.0001:
            break

        previous_loss = current_loss

    return b, m


months = [1,2,3,4,5,6,7,8,9,10,11,12]

revenue = [52,74,79,95,115,110,129,126,147,146,156,184]


b = 0
m = 0

b, m = gradient_descent(months, revenue, b, m)

print("\nFinal values:")
print("b =", b)
print("m =", m)
