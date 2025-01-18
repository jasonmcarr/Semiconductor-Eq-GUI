import tkinter as tk
from tkinter import ttk, messagebox
from ttkbootstrap import Style
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import inspect
import Semiconductor_Equations as module  # Dynamically inspect this module
import re

# Updated function to render LaTeX equations
def render_latex_equation(latex_code, parent_frame):
    for widget in parent_frame.winfo_children():  # Clear previous canvas
        widget.destroy()
    fig, ax = plt.subplots(figsize=(5, 0.8))
    ax.text(0.5, 0.5, f"${latex_code}$", fontsize=14, ha="center", va="center")
    ax.axis("off")
    canvas = FigureCanvasTkAgg(fig, master=parent_frame)
    canvas.get_tk_widget().pack(fill="both", expand=True)
    canvas.draw()

# Function to extract all functions and metadata from the module
def get_functions_metadata(module):
    metadata = {}
    for name, func in inspect.getmembers(module, inspect.isfunction):
        doc = func.__doc__ or ""
        lines = doc.strip().split("\n") if doc else []
        description = lines[0] if lines else "No description available."
        params = []
        param_section = False
        for line in lines:
            line = line.strip()
            if line.startswith("Parameters"):
                param_section = True
            elif line.startswith("Returns"):
                param_section = False
            elif param_section and ":" in line:
                parts = line.split(":")
                param_name = parts[0].strip()
                param_desc = parts[1].strip()
                params.append((param_name, "", param_desc))
        metadata[name] = {
            "description": description,
            "params": params,
            "function_reference": func,
            "unit": ""
        }
    return metadata

# GUI Logic
def build_gui(functions_metadata):
    style = Style(theme="flatly")
    root = style.master
    root.title("Semiconductor Equations Calculator")
    root.geometry("800x700")
    root.rowconfigure(1, weight=1)
    root.columnconfigure(0, weight=1)

    ttk.Label(root, text="Semiconductor Equations Calculator", font=("Helvetica", 16, "bold")).grid(row=0, column=0, pady=10, sticky="nwe")

    # Function selector with search capability
    ttk.Label(root, text="Select Function:").grid(row=1, column=0, padx=10, sticky="w")
    function_var = tk.StringVar()
    function_menu = ttk.Combobox(root, textvariable=function_var, values=list(functions_metadata.keys()), state="normal")
    function_menu.grid(row=2, column=0, padx=10, pady=5, sticky="ew")

    # Enable typing to search
    function_menu.bind("<KeyRelease>", lambda event: filter_functions(function_menu, list(functions_metadata.keys())))

    # Description box with scrollbar
    description_frame = ttk.Frame(root)
    description_frame.grid(row=3, column=0, padx=10, pady=5, sticky="nsew")
    root.rowconfigure(3, weight=1)
    description_text = tk.Text(description_frame, wrap="word", height=5, state="disabled", bg=style.colors.bg)
    description_text.pack(side="left", fill="both", expand=True)
    description_scrollbar = ttk.Scrollbar(description_frame, orient="vertical", command=description_text.yview)
    description_scrollbar.pack(side="right", fill="y")
    description_text.config(yscrollcommand=description_scrollbar.set)

    # Parameter fields
    param_entries = {}

    def filter_functions(combobox, functions):
        search_term = combobox.get().lower()
        filtered = [f for f in functions if search_term in f.lower()]
        combobox["values"] = filtered

    def update_parameters(*args):
        # Clear previous fields
        for widget in root.grid_slaves():
            if int(widget.grid_info()["row"]) > 4:
                widget.destroy()

        param_entries.clear()
        selected_function = function_var.get()

        if selected_function in functions_metadata:
            metadata = functions_metadata[selected_function]

            # Update description
            description_text.config(state="normal")
            description_text.delete("1.0", tk.END)
            description_text.insert("1.0", metadata["description"])
            description_text.config(state="disabled")

            # Display parameters
            params_frame = ttk.Frame(root)
            params_frame.grid(row=5, column=0, padx=10, pady=5, sticky="nsew")
            for param_name, unit, description in metadata["params"]:
                ttk.Label(params_frame, text=f"{param_name} ({unit}):").grid(sticky="w", padx=5, pady=2)
                entry = ttk.Entry(params_frame, width=10)
                entry.grid(sticky="ew", padx=5, pady=2)
                params_frame.columnconfigure(1, weight=1)
                param_entries[param_name] = entry

            # Add Calculate button
            ttk.Button(params_frame, text="Calculate", command=calculate).grid(column=0, columnspan=2, pady=10)

    def parse_scientific_notation(value):
        """Parses input to handle scientific notation."""
        try:
            # Replace common scientific notation patterns and convert to float
            return float(re.sub(r'[eE]', 'e', value))
        except ValueError:
            raise ValueError(f"Invalid number format: {value}")

    def calculate():
        selected_function = function_var.get()
        if selected_function in functions_metadata:
            func = functions_metadata[selected_function]["function_reference"]
            args = []
            invalid_fields = []
            try:
                for param_name, entry in param_entries.items():
                    value = entry.get().strip()
                    if not value:
                        invalid_fields.append(param_name)
                    else:
                        try:
                            args.append(parse_scientific_notation(value))
                        except ValueError:
                            invalid_fields.append(param_name)

                if invalid_fields:
                    messagebox.showerror(
                        "Input Error", 
                        f"Please enter valid numbers for the following parameters: {', '.join(invalid_fields)}."
                    )
                    return

                result = func(*args)

                # Display result in a popup and allow copying
                result_window = tk.Toplevel(root)
                result_window.title("Result")
                formatted_result = f"{result:.6g} {functions_metadata[selected_function]['unit']}"
                ttk.Label(result_window, text=f"Result: {formatted_result}").pack(padx=10, pady=10)

                # Add copy functionality
                result_entry = ttk.Entry(result_window, width=40)
                result_entry.insert(0, f"{result}")  # Copy raw value as standard notation
                result_entry.pack(pady=5)
                result_entry.config(state="readonly")

                ttk.Button(result_window, text="Copy", command=lambda: root.clipboard_append(result_entry.get())).pack(pady=5)

            except Exception as e:
                messagebox.showerror("Error", str(e))

    function_menu.bind("<<ComboboxSelected>>", update_parameters)
    root.mainloop()

# Fetch metadata from the module dynamically
functions_metadata = get_functions_metadata(module)

build_gui(functions_metadata)
