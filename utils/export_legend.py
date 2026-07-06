from paraview.simple import *

# 1. Get active view (your main plot)
view = GetActiveView()

# 2. Get the color transfer function (replace "U" with your field name)
lut = GetColorTransferFunction("u_h")

# 3. Get scalar bar (legend)
scalarBar = GetScalarBar(lut, view)

# 4. Create a new render view JUST for legend
legendView = CreateView("RenderView")

# Make background transparent (important for LaTeX later)
legendView.GradientBackground = 0
legendView.Background = [1, 1, 1]  # or [0,0,0] depending on preference

# 5. Attach scalar bar to the new view
scalarBar.Visibility = 1
scalarBar.WindowLocation = 'AnyLocation'

# IMPORTANT: ensure it is drawn in legendView
scalarBar.ScalarBarLength = 0.8
scalarBar.ScalarBarThickness = 30

# 6. Render and save ONLY legend
Render(legendView)

SaveScreenshot(
    "legend.png",
    legendView,
    ImageResolution=[300, 800],
    TransparentBackground=1
)
