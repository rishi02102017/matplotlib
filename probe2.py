import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
print("VERSION", matplotlib.__version__)
fig, ax = plt.subplots()
x = [1, 1, 2]
for comp in (False, True):
    l0 = ax.ecdf(x, complementary=comp)
    l1 = ax.ecdf(x, complementary=comp, compress=True)
    print(f"complementary={comp}")
    print("  compress=False:", list(zip(l0.get_xdata(), l0.get_ydata())))
    print("  compress=True: ", list(zip(l1.get_xdata(), l1.get_ydata())))
