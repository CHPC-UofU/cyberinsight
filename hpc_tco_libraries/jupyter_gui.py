import ipywidgets
import ipyfilechooser
import json
from IPython import display
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os
import hpc_tco_libraries.model
import textwrap


layout = ipywidgets.Layout(width='auto', height='auto')
magic_vars = {"LIFETIME"}
#SECONDS_PER_5_YEARS = 157680000
SECONDS_PER_YEAR = 31536000
CANNOT_BE_BLANK = ["avg_utilization", "cost_per_kwh","hourly_cloud_hosting_fee","system_lifetime_years","total_nodes","kwh_used"]
CANNOT_BE_ZERO =["avg_utilization","system_lifetime_years","total_nodes"]
FLOAT_FIELD = ["avg_utilization", "cost_per_kwh", "hourly_cloud_hosting_fee", "system_lifetime_years","total_nodes", 
                "kwh_used", "node_count", "wall_time"]
form_valid = 0

resource_templates = {
    "compute": {
        "template_string": "[+] new Compute resource...",
        "attributes": {
            "total_nodes": 1,
            "cost_per_kwh": 0.0,
            "hourly_cloud_hosting_fee": 0.0,
            "avg_utilization": 1.0,
            "system_lifetime_years": 5
        },
        "value": "{upfront_costs} + ({LIFETIME}*{recurring_costs})",
        "breakdown": {
            "upfront_costs": {
                "description": "",
                "label": "",
                "value": "({hardware} + {infrastructure} + {network})",
                "breakdown": {
                    "hardware": {"value": 0.0},
                    "infrastructure": {"value": 0.0},
                    "network": {"value": 0.0}
                }
            },
            "recurring_costs": {
                "description": "",
                "label": "",
                "value": "({personnel} + {idle_power})",
                "breakdown": {
                    "personnel": {"value": 0.0},
                    "idle_power": {"value": 0.0}
                }
            },
        },
        "benchmark_parameters": {
            "node_count": 1,
            "wall_time": 0,
            "kwh_used": 0.0
        }
    },
    "storage": {
        "template_string": "[+] new Storage resource...",
        "attributes": {
            "tb_capacity": 1
        },
        "value": "{upfront_costs} + ({LIFETIME}*{recurring_costs})",
        "breakdown": {
            "upfront_costs": {
                "description": "",
                "label": "",
                "value": "({hardware} + {infrastructure} + {network})"
            },
            "recurring_costs": {
                "description": "",
                "label": "",
                "value": "({personnel} + {idle_power})"
            },
        },
        "benchmark_parameters": {
            "gb_uploaded": 0,
            "gb_downloaded": 0,
            "wall_time": 0,
            "kwh_used": 0
        }
    }
}

class Importer:
    """
    An informative ipywidgets GUI for importing and troubleshooting TCO data files (JSON).
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.current_error = None
        self.imported_data = None
        
        # File chooser
        self.select = ipyfilechooser.FileChooser(
            title="Import TCO dataset (.json):"
        )

        def _validate_selection(*args):
            self.current_error = None
            self.status_details.value = ""
            self.status_note.value = ""
            self.status.layout.display = 'none'
            redfont = "<font color='red'>"
            try:
                self.imported_data = hpc_tco_libraries.model.load_dataset(self.select.selected)
                hpc_tco_libraries.model.validate_dataset(self.imported_data["data"])
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                self.status_note.value = f"{redfont}File could not be parsed as JSON."
                self.current_error = e
            #except (schema_exception._Error) as e:

            # Update confirm button
            if self.current_error is None:
                self.status.layout.display = 'none'
                self.confirm.disabled = False
                self.confirm.button_style = "success"
                self.confirm.icon = "check"
                self.confirm.tooltip = f"Data from 'blah' will be loaded for analysis."
            else:
                self.status.layout.display = 'block'

        self.select.register_callback(_validate_selection)

        # Status information       
        self.status = ipywidgets.VBox()
        self.status.layout.display = 'none'

        self.status_note = ipywidgets.HTML(
            value=""
        )

        self.status_details = ipywidgets.HTML(
            value=""
        )

        self.status_button = ipywidgets.Button(
            description='Show details',
            button_style='info',
            tooltip='Print the error associated with the selected file, for debugging purposes.',
            icon='bug'
        )

        def _toggle_status_details(*args):
            if self.status_details.value == "":
                self.status_details.value = f"<pre>{self.current_error.__class__.__name__}: {self.current_error}</pre>"
                self.status_button.description = "Hide details"
            else:
                self.status_details.value = ""
                self.status_button.description = "Show details"

        self.status_button.on_click(_toggle_status_details)
        self.status.children = [self.status_note, self.status_button, self.status_details]

        # Confirm import button       
        self.confirm = ipywidgets.Button(
            description='Import',
            disabled=True,
            button_style='danger',
            tooltip='Please select a valid TCO dataset file (.json)',
            icon='times'
        )
        def _import_data(*args):
            if self.imported_data:
                self.display()
                print("SUCCESS: Imported ", ", ".join(self.imported_data["data"].keys()))
                    
                self.dataset["data"].update(self.imported_data["data"])

        self.confirm.on_click(_import_data)
        
        # UI Layout
        self.ui = ipywidgets.VBox([self.select, self.status, self.confirm])
            

    def display(self):
        display.clear_output(wait=True)
        display.display(self.ui)


class ChartViewer:
    def __init__(self, dataset):
        self.dataset = dataset
        self.benchmarks = dict()
        self.select = ipywidgets.VBox()
        self.select_menu = ipywidgets.SelectMultiple(
            options=(),
            value=(),
            description='Data',
            disabled=False,
            layout=ipywidgets.Layout(width='auto')
        )
        self.select_title = ipywidgets.Text(
            description="Chart title",
            value="Cloud TCO analysis",
            placeholder="Enter a title",
        )
        self.select_confirm = ipywidgets.Button(
            description='Compare',
            button_style='success',
            tooltip='Create a chart comparing the cost breakdowns',
            icon='check'
        )
        self.select_confirm.on_click(self.display)

        self.select.children = [self.select_menu, self.select_title, self.select_confirm]

        self.fig, self.ax = plt.subplots()
        self.save_image = ipywidgets.HBox()
        self.save_image_name = ipywidgets.Text(
            value="",
            placeholder="Choose a filename (.png will be appended)"
        )
        self.save_image_confirm = ipywidgets.Button(
            description='Save image',
            button_style='success',
            tooltip='Save chart as an image',
            icon='floppy-o'
        )
        self.save_image_msg = ipywidgets.HTML()
        def _save_image(*args):
            exportdir = "exports/"
            if not os.path.exists(exportdir):
                os.makedirs(exportdir)
            self.fig.savefig(exportdir + str(self.save_image_name.value))
            self.save_image_msg.value = str(self.save_image_name.value) + " has been saved in " +exportdir + " directory."
        self.save_image_confirm.on_click(_save_image)
        self.save_image.children = (self.save_image_name, self.save_image_confirm)

        self.charts = ipywidgets.VBox()
        self.ui = ipywidgets.VBox([self.select, self.charts, self.save_image,self.save_image_msg])

    def update(self):
        self.benchmarks = {f"{bm['name']} | {k}": bm for k,v in self.dataset["data"].items() for bm in v["benchmarks"]}
        self.select_menu.options = tuple(sorted(self.benchmarks.keys()))
        self.select_menu.rows=len(self.select_menu.options)+1


    def display(self, *args):
        self.save_image_msg.value = ""
        self.update()
        display.clear_output(wait=True)

        # Unused:
        def plot_capex_breakdown(**kwargs):
            entries = tuple(self.select_menu.value)

            categories = set()
            for k in entries:
                v = self.dataset["data"][k]
                categories.update(v["breakdown"].keys())
                
            cat = tuple(categories)
            values = {k: [0]*len(entries) for k in cat}

            sums = [0]*len(entries)
            offsets = {k: sums[:] for k in cat}

            for k in cat:
                offsets[k] = sums[:]
                for i in range(len(entries)):
                    try:
                        d = self.dataset["data"][entries[i]]["breakdown"][k]
                        values[k][i] = hpc_tco_libraries.model.evaluate_cost(d, **kwargs)
                    except KeyError:
                        pass
                    sums[i] += values[k][i]

            width = 0.35       # the width of the bars: can also be len(x) sequence

            self.ax.clear()
            for k in cat:
                self.ax.bar(entries, np.array(values[k]), width=width, bottom=offsets[k], label=k)

            self.ax.set_ylabel('Cost ($)')
            self.ax.set_title('Cloud TCO analysis')
            self.ax.legend()

            #plt.show()

        def plot_benchmarks(**kwargs):
            benchmarks = tuple(self.select_menu.value)
            cat = ("TCO", "Power fees", "Cloud fees")
            values = {k: [] for k in cat}

            sums = [0]*len(benchmarks)
            offsets = {k: sums[:] for k in cat}
            
            def getres(benchmark):
                return benchmark.split(' | ')[-1]

            for b in benchmarks:
                res = getres(b)
                efficiency = self.dataset["data"][res]["attributes"]["avg_utilization"]
                lifetime_seconds = self.dataset["data"][res]["attributes"]["system_lifetime_years"] * SECONDS_PER_YEAR
                node_scaling = self.benchmarks[b]["node_count"] / self.dataset["data"][res]["attributes"]["total_nodes"]
                values["TCO"].append((self.benchmarks[b]["wall_time"] / lifetime_seconds) * hpc_tco_libraries.model.evaluate_cost(self.dataset["data"][res], LIFETIME=float(self.dataset["data"][res]["attributes"]["system_lifetime_years"]), **kwargs) * node_scaling / efficiency)
                values["Power fees"].append(self.benchmarks[b]["kwh_used"] * self.dataset["data"][res]["attributes"]["cost_per_kwh"])
                values["Cloud fees"].append((self.benchmarks[b]["wall_time"] / 3600) * self.dataset["data"][res]["attributes"]["hourly_cloud_hosting_fee"] * self.benchmarks[b]["node_count"] ) 

            for k in cat:
                offsets[k] = sums[:]
                for i in range(len(benchmarks)):
                    sums[i] += values[k][i]

            self.ax.clear()
            for k in cat:
                self.ax.bar(benchmarks, np.array(values[k]), width=0.2, bottom=offsets[k], label='\n'.join(textwrap.wrap(k, 16)) )

            #self.ax.set_ylabel('Cost ($)')
            fmt = '${x:,.2f}'
            tick = mtick.StrMethodFormatter(fmt)
            self.ax.yaxis.set_major_formatter(tick) 
            self.ax.set_title(self.select_title.value)
            self.ax.legend()
            plt.setp(self.ax.get_xticklabels(), rotation=13, horizontalalignment='right', fontsize='x-small')
            plt.tight_layout()

            display.display(self.fig)
            print("TOTAL BENCHMARK COSTS")
            for k,v in zip(benchmarks, sums):
                print(k, '=', '${:,f}'.format(v))

        if len(self.select_menu.value) > 0:
            #interactive_plot = ipywidgets.interactive(plot_benchmarks, year=ipywidgets.IntSlider(min=1, max=10, continuous_update=False))
            interactive_plot = ipywidgets.interactive(plot_benchmarks)
            self.charts.children = [interactive_plot]
        else:
            self.charts.children = []

        display.display(self.ui)


class Form:
    def __init__(self, dataset):
        self.dataset = dataset

        self.workbench = dict()
        self.unsaved = dict()

        self.select = ipywidgets.VBox()

        # The menu where TCO entries can be selected
        self.select_menu = ipywidgets.SelectMultiple(
            options=[v["template_string"] for v in resource_templates.values()] + list(self.dataset["data"].keys()),
            value=(),
            description="",
            disabled=False,
            layout=layout
        )
        self.select_menu.value = (self.select_menu.options[0],)

        # The text box associated with the menu
        self.select_name = ipywidgets.Text(
            value=self.select_menu.value[0],
            placeholder="Choose a new or existing dataset name",
            disabled=False,
            layout=ipywidgets.Layout(width="100%")
        )

        def _observed_SelectMultiple(widg):
            if len(widg["new"]) >1:
                self.select_save.disabled = True
                self.select_load.disabled = True
            elif len(widg["new"]) == 1:
                self.select_name.value = widg["new"][0]
            deleteFlag = True
            for file in widg["new"]:
                if file.startswith('[+]'):
                    deleteFlag = False
            if not deleteFlag:
                self.select_delete.disabled = True
            else:
                self.select_delete.disabled = False
        self.select_menu.observe(_observed_SelectMultiple,"value")
        
        # Changes in selection will update the text field
        # ipywidgets.dlink((self.select_menu, 'value'), (self.select_name, 'value'))
        def _edit_name(change):
            """Prevent reserved or illegal names from being used as new entry names"""
            if len(change["new"]) < 1 or change["new"].startswith('[+]'):
                self.select_save.disabled = True
            elif self.workbench:
                self.select_save.disabled = False

            if change["new"] not in self.select_menu.options:
                self.select_load.disabled = True
            else:
                self.select_load.disabled = False


        self.select_name.observe(_edit_name, "value")

        def deepcopy(d):
            """Thread-safe dictionary deep copy"""
            return json.loads(json.dumps(d))

        # Save button (overwrites entry with select_name value)
        self.select_save = ipywidgets.Button(
            description='Save',
            button_style='success',
            tooltip='Save or overwrite the data',
            icon='floppy-o',
            disabled = True,
            layout=layout
        )
        self.save_msg = ipywidgets.HTML()

        def _save(*args):
            global form_valid
            if form_valid > 0:
                self.save_msg.value = self.select_name.value + " <strong style='color: red;'>CAN NOT</strong> be saved, Please check the form validation."
            else:
                self.workbench["data"]["benchmarks"] = [b for b in self.workbench["data"]["benchmarks"] if b["name"] != ""]
                self.dataset["data"][self.select_name.value] = deepcopy(self.workbench)["data"]
                _load()
                self.save_msg.value = self.select_name.value + " has been saved to memory."
                self.update()
        self.select_save.on_click(_save)

        # Load button
        self.select_load = ipywidgets.Button(
            description='Load',
            button_style='success',
            tooltip='Load the data into a form',
            icon='folder-open-o',
            layout=layout
        )
        def _load(*args):
            global form_valid
            form_valid = 0
            self.save_msg.value =""
            if self.workbench:
                self.unsaved = deepcopy(self.workbench)
            if self.select_name.value in self.dataset["data"]:
                self.workbench = {"name": self.select_name.value, "data": deepcopy(self.dataset["data"][self.select_name.value])}  # Thread-safe deep copy
                self.update()
                self.select_save.disabled = False
            elif self.select_name.value in [v["template_string"] for v in resource_templates.values()]:
                for k,v in resource_templates.items():
                    if self.select_name.value == v["template_string"]:
                        self.workbench = {
                            "name": f"Untitled {k}",
                            "data": {
                                "type": k,
                                "value": v["value"],
                                "breakdown": json.loads(json.dumps(v["breakdown"])),
                                "description": "",
                                "attributes": {k: val for k, val in v["attributes"].items()},
                                "benchmarks": list()
                                }
                            }
                        self.update()
                        self.select_save.disabled = True
        self.select_load.on_click(_load)

        # Delete the named entry
        self.select_delete = ipywidgets.Button(
            description='Delete',
            button_style='danger',
            tooltip='Delete the selected entries',
            icon='trash',
            disabled=True,
            layout=layout
        )
        def _delete(*args):
            self.save_msg.value = ""
            for v in  self.select_menu.value:
                if v in self.dataset["data"].keys():
                    del self.dataset["data"][v]
            self.update()
        self.select_delete.on_click(_delete)

        # Assemble the GUI from the components
        self.form = ipywidgets.VBox()
        self.form.layout.display = 'none'
        self.attributes_header = ipywidgets.HTML(value="<h4>Attributes</h4>")
        self.attributes_fields = ipywidgets.VBox()
        self.cost_header = ipywidgets.HTML(value="<h4>Costs</h4>")
        self.cost_fields = ipywidgets.VBox()
        self.benchmark_header = ipywidgets.HTML(value="<h4>Benchmarks</h4>")
        self.benchmark_fields = ipywidgets.VBox()
        self.benchmark_create = ipywidgets.Button(
            description='New benchmark',
            button_style='success',
            tooltip='Create a new benchmark',
            icon='plus'
        )
        def _new_benchmark(*args):
            # add new benchmark with appropriate schema to the workbench
            for k,v in resource_templates.items():
                if self.workbench["data"]["type"] == k:
                    benchmark = {k:val for k,val in v["benchmark_parameters"].items()}
                    benchmark.update({"description": ""})
                    self.workbench["data"]["benchmarks"].append(benchmark)
                    self.benchmark_fields.children = [Form.AttrForm(v) for v in self.workbench["data"]["benchmarks"]]
        self.benchmark_create.on_click(_new_benchmark)

        self.form.children = [
            self.attributes_header,
            self.attributes_fields,
            self.cost_header,
            self.cost_fields,
            self.benchmark_header,
            self.benchmark_fields,
            self.benchmark_create
            ]

        self.filler_text = ipywidgets.HTML(value="Select and load an option above.")
        self.select.children = [self.select_menu, self.filler_text, self.select_load, self.select_delete,self.select_name, self.select_save, self.save_msg]

        self.ui = ipywidgets.VBox([self.select, self.form])

        self.update()
            
    def update(self):
        last_name = self.select_name.value
        # self.select_menu.options = tuple([v["template_string"] for v in resource_templates.values()] + list(self.dataset["data"].keys()))
        if self.select_name.value in self.select_menu.options:
            self.select_menu.value = (self.select_name.value,)

        if self.workbench:
            root_entry = self.workbench_to_form()
            root_entry._submit_value()
            root_entry.name.description_tooltip = "Select or write a new name in the above menu and press 'Save' to modify this field."
            self.cost_fields.children = [root_entry]

            self.attributes_fields.children = [Form.AttrForm(self.workbench["data"]["attributes"], singleton=True)]

            if "benchmarks" in self.workbench["data"]:
                # Clean up deleted entries
                self.workbench["data"]["benchmarks"] = [b for b in self.workbench["data"]["benchmarks"] if b["name"] != ""]
                self.benchmark_fields.children = [Form.AttrForm(v) for v in self.workbench["data"]["benchmarks"]]
            else:
                self.benchmark_fields.children = ()

        if self.form.layout.display == 'none':
            if len(self.cost_fields.children) > 0:
                self.form.layout.display = 'block'
                self.filler_text.layout.display = 'none'

        self.select_name.value = last_name

    def display(self):
        display.clear_output(wait=True)
        display.display(self.ui)

    class CostField(ipywidgets.VBox):
        def __init__(self, name, data):
            super().__init__()
            self.dirty = False
            self.data = data
            self.interface = ipywidgets.HBox()
            self.layout.margin = "0px 0px 0px 32px"  # Left indent for visual nesting
            self.name = ipywidgets.Text(
                value=name,
                placeholder="Name",
                disabled=True,
                layout=ipywidgets.Layout(width="128px")
            )
            self.value = ipywidgets.Text(
                value=str(data["value"]),
                placeholder="Value",
                disabled=False,
                layout=ipywidgets.Layout(width="100%")
            )
            self.last_value = self.value.value
            self.value.on_submit(self._submit_value)

            def _edit_value(change):
                if self.dirty:
                    if self.last_value == change["new"]:
                        self.dirty = False
                        self.value.layout.border = ""
                else:
                    self.dirty = True
                    self.value.layout.border = "dashed yellow"
            self.value.observe(_edit_value, "value")

            self.collapse_button = ipywidgets.Button(
                description='',
                button_style='',
                tooltip='Toggle children (if applicable)',
                layout = ipywidgets.Layout(width="32px")
            )
            def _toggle_children(*args):
                if self.collapsed:
                    self.collapsed = False
                    self.collapse_button.icon = 'chevron-down'
                    self._children.layout.display = 'block'
                else:
                    self.collapsed = True
                    self.collapse_button.icon = 'chevron-right'
                    self._children.layout.display = 'none'
            self.collapse_button.on_click(_toggle_children)

            self._children = ipywidgets.VBox()
            self.collapsed = False
            _toggle_children()

            self.interface.children = [self.collapse_button, self.name, self.value]
            self.children = [self.interface, self._children]
        
        
        def _submit_value(self, *args):
            breakdown = set(hpc_tco_libraries.model.re_variables.findall(self.value.value))
            # Remove "magic" variables like "year"
            if breakdown - magic_vars:
                children = [field for field in self._children.children if field.name.value in breakdown]
                if "breakdown" not in self.data:
                    self.data["breakdown"] = dict()
                else:
                    for removed_entry in set(self.data["breakdown"].keys()) - breakdown:
                        del self.data["breakdown"][removed_entry]
                for entry in breakdown - magic_vars:
                    if entry not in self.data["breakdown"]:
                        self.data["breakdown"][entry] = {"label": "", "description": "", "value": 0.0}
                        field = Form.CostField(entry, self.data["breakdown"][entry])
                        if "breakdown" not in field.data:
                            field.collapse_button.disabled = True
                            field.collapse_button.icon = "circle-o"
                        children.append(field)
                self._children.children = children
                self.collapsed = False
                self.collapse_button.icon = 'chevron-down'
                self._children.layout.display = 'block'
                self.collapse_button.disabled = False
                self.data["value"] = self.value.value
            elif breakdown.issubset(magic_vars):
                self.data["value"] = self.value.value
                self.collapse_button.disabled = True
                self.collapse_button.icon = "circle-o"
                self._children.children = ()
            else:
                try:
                    self.data["value"] = float(self.value.value)
                    self.collapse_button.disabled = True
                    self.collapse_button.icon = "circle-o"
                    self._children.children = ()
                except ValueError:
                    # Handle error
                    return
            self.dirty = False
            self.value.layout.border = ""
            self.last_value = self.value.value
        
    class AttrForm(ipywidgets.VBox):
        def __init__(self, data, singleton=False):
            super().__init__()
            self.dirty = False
            self.data = data
            self.interface = ipywidgets.HBox()
            if "name" in self.data:
                if self.data["name"] == "":
                    return
                self.name = ipywidgets.Text(
                    value=self.data["name"],
                    placeholder="Name",
                    disabled=False,
                    layout=ipywidgets.Layout(width="128px")
                )
                self.name.dirty = False
            else:
                self.data["name"] = ""
                self.name = ipywidgets.Text(
                    value="",
                    placeholder="untitled",
                    disabled=False,
                    layout=ipywidgets.Layout(width="128px")
                )
                self.name.layout.border = "dashed red"
                self.name.dirty = True
                global form_valid
                form_valid += 1

            def _rename(change):
                self.data["name"] = change["new"].strip()
                global form_valid

                if self.data["name"] == "":
                    if not self.name.dirty:
                        self.name.layout.border = "dashed red"
                        self.name.dirty = True
                        form_valid += 1
                else:
                    if self.name.dirty:
                        self.name.layout.border = ""
                        self.name.dirty = False
                        form_valid -= 1

                # save to main data
                # dict to array

            self.name.observe(_rename, "value")
            self.fields = ipywidgets.VBox()
            self._fields = []

            self.children = [self.interface, self.fields]

            def _submit_fvalue(widget):
                validFlag = True
                global form_valid
                try:
                    self.data[widget.name] = float(widget.value)
                except:
                    if widget.name in FLOAT_FIELD:
                        widget.placeholder = "Can not convert to float"
                        widget.value = ""
                        validFlag = False
                    else:
                        self.data[widget.name] = widget.value
                if validFlag and widget.name == "avg_utilization" and self.data[widget.name] > 1:
                    widget.placeholder = "Utilization can not exceed 1 (100%)"
                    widget.value = ""
                    validFlag = False
                if validFlag and widget.name in CANNOT_BE_BLANK and len(str(self.data[widget.name]).strip()) < 1:
                    widget.placeholder = "Can not be blank"
                    widget.value = ""
                    validFlag = False
                if validFlag and widget.name in CANNOT_BE_ZERO and self.data[widget.name] == 0:
                    widget.placeholder = "Can not be zero"
                    widget.value = ""
                    validFlag = False
                if validFlag:
                    widget.last_entry = widget.value
                    widget.dirty = False
                    if widget.layout.border == "dashed yellow":
                        form_valid -= 1
                    elif widget.layout.border == "dashed red":
                        form_valid -= 2
                    widget.layout.border = ""
                    widget.placeholder = ""
                else:
                    widget.dirty = True
                    if widget.layout.border != "dashed red":
                        form_valid += 1
                        widget.layout.border = "dashed red"

            def _edit_value(change):
                global form_valid
                widget = change["owner"]
                if not widget.dirty:
                    widget.dirty = True
                    form_valid += 1
                    widget.layout.border = "dashed yellow"

            for field, value in self.data.items():
                if field == "name":
                    continue
                interface = ipywidgets.HBox()
                interface.layout.margin = "0px 0px 0px 32px"
                fname = ipywidgets.Text(
                    value=field,
                    placeholder="",
                    disabled=True,
                    layout=ipywidgets.Layout(width="128px")
                )
                fvalue = ipywidgets.Text(
                    value=str(value),
                    placeholder="",
                    disabled=False,
                )
                fvalue.name = fname.value
                fvalue.last_entry = fname.value
                fvalue.dirty = False
                fvalue.observe(_edit_value, "value")
                fvalue.on_submit(_submit_fvalue)
                interface.children = [fname, fvalue]
                self._fields.append(interface)

            self.fields.children = self._fields

            self.collapsed = False
            self.collapse_button = ipywidgets.Button(
                description='',
                button_style='',
                icon='chevron-down',
                tooltip='Toggle fields',
                layout = ipywidgets.Layout(width="32px")
            )
            def _toggle_children(*args):
                if self.collapsed:
                    self.collapsed = False
                    self.collapse_button.icon = 'chevron-down'
                    self.fields.layout.display = 'block'
                else:
                    self.collapsed = True
                    self.collapse_button.icon = 'chevron-right'
                    self.fields.layout.display = 'none'
            self.collapse_button.on_click(_toggle_children)

            self.delete_button = ipywidgets.Button(
                description='',
                button_style='danger',
                icon='trash',
                tooltip='Delete benchmark',
                layout = ipywidgets.Layout(width="32px")
            )
            def _delete(*args):
                self.deleted = True
                self.layout.display = 'none'
                self.data["name"] = ""
            self.delete_button.on_click(_delete)

            if singleton:
                self.delete_button.layout.display = 'none'
                self.name.value = "Attributes"
                self.name.disabled = True
            self.interface.children = [self.collapse_button, self.name, self.delete_button]


    def workbench_to_form(self):
        def _entry_to_field(name, data):
            field = Form.CostField(name, data)
            if "breakdown" in data:
                children = list()
                for k, v in data["breakdown"].items():
                    children.append(_entry_to_field(k, v))
                field._children.children = children
            else:
                field.collapse_button.disabled = True
                field.collapse_button.icon = "circle-o"
            return field

        root = _entry_to_field(self.workbench["name"], self.workbench["data"])
        root.layout.margin = "0px"

        return root


class Exporter:
    def __init__(self, dataset):
        self.dataset = dataset

        self.select_entries = ipywidgets.SelectMultiple(
            options=list(self.dataset["data"].keys()),
            value=(),
            description="",
            rows=len(list(self.dataset["data"].keys()))+1,
            layout=layout
        )
        exportdir = "exports/"
        if not os.path.exists(exportdir):
            os.makedirs(exportdir)
        self.filename = ipyfilechooser.FileChooser(
            title="Choose a file to overwrite or type a new filename to export TCO dataset (.json):",
        )
        self.filename.default_path=os.path.abspath(exportdir)

        def _validate_filename(*args):
            self.export_msg.value = ""
            if self.filename.selected_filename.strip() == "":
                self.export_button.disabled = True
                self.export_button.button_style = "danger"
                self.export_button.icon = "times"
                self.export_button.tooltip = f"Please enter a valid filename."
                self.export_msg.value = "<strong style='color: red;'>Please enter a valid filename</strong>."

            else:
                self.export_button.disabled = False
                self.export_button.button_style = "success"
                self.export_button.icon = "check"
                self.export_button.tooltip = f"Click to export data."
                self.newfile = self.filename.selected

        self.filename.register_callback(_validate_filename)

        self.export_button = ipywidgets.Button(
            description='Export',
            disabled=True,
            button_style='danger',
            tooltip='Please enter a valid filename',
            icon='times'
        )
        self.export_button.on_click(self._export)

        self.obfuscate_finances = ipywidgets.Checkbox(
            value=False,
            description="Obfuscate financial details"
        )
        self.export_msg =  ipywidgets.HTML()
        #self.refresh_button = ipywidgets.Button(
        #    description='Refresh',
        #    disabled=False,
        #    button_style='info',
        #    tooltip='Refresh the menu.',
        #    icon='refresh'
        #)
        #self.refresh_button.on_click(self.display)

        self.ui = ipywidgets.VBox([self.select_entries, self.obfuscate_finances,self.filename, self.export_button,self.export_msg])

    def display(self, *args):
        display.clear_output(wait=True)
        display.display(self.ui)

    def _export(self, *args):
        # if not self.dataset_name.value:
        #     return

        result = {"data": dict()}
        if self.select_entries.value:
            for entry in tuple(self.select_entries.value):
                result["data"][entry] = self.dataset["data"][entry]
        else:  # export everything rather than nothing
            for entry in tuple(self.select_entries.options):
                result["data"][entry] = self.dataset["data"][entry]

        result_copy = json.loads(json.dumps(result))

        # Obfuscate finances
        if self.obfuscate_finances.value == True:
            for entry in result_copy["data"].values():
                entry["value"] = hpc_tco_libraries.model.evaluate_cost(entry, LIFETIME=float(entry["attributes"]["system_lifetime_years"]))
                if "breakdown" in entry:
                    del entry["breakdown"]

        if self.newfile:
            self.newfile = self.newfile.strip()
            if (len(self.newfile) > 4 and self.newfile[-5:] != '.json') or len(self.newfile) < 5: 
                self.newfile = self.newfile +'.json'
            with open(self.newfile, "w") as outfile:
                json.dump(result_copy, outfile, sort_keys=True, indent=4)
                self.export_msg.value = "Data has been exported to " + self.newfile + "."
        else:
            self.export_msg.value = ""
