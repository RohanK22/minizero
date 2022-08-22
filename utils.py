from IPython.display import display

def show_svg(svg):
    display(svg)
    
# =============================================================================
# def get_compute_device():
#     compute_device = None
#     # detect gpu/cpu device to use
#     if torch.backends.cuda.is_built():
#         compute_device = torch.device('cuda:0') # 0th CUDA device
#     if torch.backends.mps.is_available():
#         compute_device = torch.device('mps') # For Apple silicon
#     else:
#         compute_device = torch.device("cpu") # Use CPU if no GPU
#     return torch.device("cpu")#compute_device# compute_device
# =============================================================================
