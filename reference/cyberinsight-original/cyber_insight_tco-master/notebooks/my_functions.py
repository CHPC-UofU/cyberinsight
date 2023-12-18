from datetime import datetime as dt
from IPython.display import display, HTML
from ipywidgets import widgets, interactive, Layout
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
from plotly.tools import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from xlrd import XLRDError

import my_DB_functions as myDB

init_notebook_mode()
MAX_YEARS = 7

COLORS_10 = ['rgb(165,0,38)',
         'rgb(215,48,39)',
         'rgb(244,109,67)',
         'rgb(253,174,97)',
         'rgb(254,224,144)',
         'rgb(224,243,248)',
         'rgb(171,217,233)',
         'rgb(116,173,209)',
         'rgb(69,117,180)',
         'rgb(49,54,149)']

def print_title(text, h_type="h1"):
    display(HTML("<{}>{}</{}>".format(h_type, text, h_type)))
    
def print_table(data, width=None):
    if width == None:
        table = '<table>'
    else:
        table = '<table width="{}">'.format(width)
    display(HTML(
        table +
            '<thead><tr>{}</tr></thead>'.format(''.join('<th>{}</th>'.format(item) for item in data[0])) +
            '<tbody><tr>{}</tr></tbody></table>'.format('</tr><tr>'.join('<td>{}</td>'.format('</td><td>'.join(str(_) for _ in row)) for row in data[1:]))
    ))

def display_scaling(x_vals, y_vals, num_servers_vals, resource_type, x_axis, y_axis, degree, max_x1, max_x2):
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, print_grid=False, horizontal_spacing=0.01)
    
    ####################################################################
    X = x_vals
    y = y_vals
    
    X_reshape = X.reshape(-1, 1)

    poly = PolynomialFeatures(degree=degree)
    X_reshape_poly = poly.fit_transform(X_reshape)
    
    lr = LinearRegression()
    lr.fit(X_reshape_poly, y)
    
    X_new = np.arange(1, int(max_x1)+1)
    X_new_reshape = X_new.reshape(-1, 1)
    X_new_reshape_poly = poly.fit_transform(X_new_reshape)
    y_new_pred = lr.predict(X_new_reshape_poly)
    
    trace0 = go.Scatter(x=X_new, y=y_new_pred, mode="lines", name=x_axis)
    trace1 = go.Scatter(x=X, y=y, mode="markers", marker=dict(size=15))
    fig.append_trace(trace0, 1, 1)
    fig.append_trace(trace1, 1, 1)
    
    ####################################################################
    X = num_servers_vals
    y = y_vals
    
    X_reshape = X.reshape(-1, 1)

    poly = PolynomialFeatures(degree=degree)
    X_reshape_poly = poly.fit_transform(X_reshape)
    
    lr = LinearRegression()
    lr.fit(X_reshape_poly, y)
    
    X_new = np.arange(1, int(max_x2)+1)
    X_new_reshape = X_new.reshape(-1, 1)
    X_new_reshape_poly = poly.fit_transform(X_new_reshape)
    y_new_pred = lr.predict(X_new_reshape_poly)
    
    trace2 = go.Scatter(x=X_new, y=y_new_pred, mode="lines", name="Number of {}".format("Instances" if resource_type == "cloud" else "Nodes"))
    trace3 = go.Scatter(x=X, y=y, mode="markers", marker=dict(size=15))
    fig.append_trace(trace2, 1, 2)
    fig.append_trace(trace3, 1, 2)
    
    ####################################################################
    fig["layout"]["title"] = "Performance Scaling Functions"
    fig["layout"]["showlegend"] = False
    fig["layout"]["yaxis"].update(title=y_axis)
    fig["layout"]["hovermode"] = "closest"
    fig["layout"]["xaxis1"].update(title=x_axis, showspikes=True)
    fig["layout"]["xaxis2"].update(title="Number of {}".format("Instances" if resource_type == "cloud" else "Nodes"), showspikes=True)
    fig["layout"]["yaxis"].update(showspikes=True)
    iplot(fig, show_link=False)
    
    print_title("Hover over the line to see the corresponding X and Y values of a point of interest.", "h3")

def display_multiple_scaling(df, record_id_list, df_xy_list, **params):
    for record_id, df_xy in zip(record_id_list, df_xy_list):
        sys_type = df["Type"][record_id]
        org_name = df["Org"][record_id]
        sys_name = df["System"][record_id]
        x_unit = df["X"][record_id]
        y_unit = df["Y"][record_id]
        degree = params["degree_{}".format(record_id)]
        
        if len(df_xy) == 1:
            df_xy = df_xy.append({"x": 0, "y":0}, ignore_index=True)
            
        x_vals = np.array(df_xy.x)
        y_vals = np.array(df_xy.y)
        # num_servers_vals = np.array(df_xy.num_servers)
        
        max_x1 = max(x_vals) * 3
        # max_x2 = max(num_servers_vals) * 3
        
        # fig = make_subplots(rows=1, cols=2, shared_yaxes=True, print_grid=False, horizontal_spacing=0.01)
        fig = make_subplots(rows=1, cols=1, shared_yaxes=True, print_grid=False, horizontal_spacing=0.01)

        ####################################################################
        X = x_vals
        y = y_vals

        X_reshape = X.reshape(-1, 1)

        poly = PolynomialFeatures(degree=degree)
        X_reshape_poly = poly.fit_transform(X_reshape)

        lr = LinearRegression()
        lr.fit(X_reshape_poly, y)

        X_new = np.arange(1, int(max_x1)+1)
        X_new_reshape = X_new.reshape(-1, 1)
        X_new_reshape_poly = poly.fit_transform(X_new_reshape)
        y_new_pred = lr.predict(X_new_reshape_poly)

        trace0 = go.Scatter(x=X_new, y=y_new_pred, mode="lines", name=x_unit)
        trace1 = go.Scatter(x=X, y=y, mode="markers", marker=dict(size=15))
        fig.append_trace(trace0, 1, 1)
        fig.append_trace(trace1, 1, 1)


        ####################################################################
        fig["layout"]["title"] = "Performance Scaling for {} {} {} (ID {})".format(org_name, sys_name, sys_type, record_id)
        fig["layout"]["showlegend"] = False
        fig["layout"]["yaxis"].update(title=y_unit)
        fig["layout"]["hovermode"] = "closest"
        fig["layout"]["xaxis"].update(title="Number of {}".format("Instances" if sys_type == "Cloud" else "Nodes"), showspikes=True)
        fig["layout"]["yaxis"].update(showspikes=True)
        fig["layout"]["height"] = 300
        iplot(fig, show_link=False)
    
    print_title("Hover over the line to see the corresponding X and Y values of a point of interest.", "h3")

def calculate_TCO_cloud(params, num_servers, years, util):
    instance_count = int(np.ceil(num_servers * (util / 100)))
    hourly_instance_cost = params["hourly_instance_cost"]
    storage_requirement = params["storage_requirement"]
    monthly_storage_cost = params["monthly_storage_cost"]
    personnel_setup_cost = params["personnel_setup_cost"]
    monthly_personnel_cost = params["monthly_personnel_cost"]
    monthly_network_transfer_cost = params["monthly_network_transfer_cost"]

    ###################################
    # Subcost calculation
    ###################################
    cloud_instance_cost = instance_count * hourly_instance_cost * 24 * 365 * years
    cloud_storage_cost = storage_requirement * monthly_storage_cost * years * 12
    personnel_cost = personnel_setup_cost + monthly_personnel_cost * years * 12
    network_cost = monthly_network_transfer_cost * years * 12
    
    return {"Instance": cloud_instance_cost, "Storage": cloud_storage_cost, "Network": network_cost, "Personnel": personnel_cost}

def calculate_TCO_onpremise(params, num_servers, years):
    node_count = num_servers
    cost_per_node = params["cost_per_node"]
    rack_infra_cost = params["rack_infra_cost"]
    power_infra_cost = params["power_infra_cost"]
    monthly_data_center_cost = params["monthly_data_center_cost"]
    monthly_power = params["monthly_power"]
    power_cost = params["power_cost"]
    personnel_setup_cost = params["personnel_setup_cost"]
    monthly_personnel_cost = params["monthly_personnel_cost"]
    
    ###################################
    # Subcost calculation
    ###################################
    server_cost = node_count * cost_per_node
    server_cost = node_count * cost_per_node
    power_consumption_cost = monthly_power * power_cost * years * 12
    data_center_space_cost = monthly_data_center_cost * years * 12
    personnel_cost = personnel_setup_cost + monthly_personnel_cost * years * 12
    
    return {"Server": server_cost, "Rack Infra": rack_infra_cost, "Power Infra": power_infra_cost, 
            "Power Consumption": power_consumption_cost, "Data Center": data_center_space_cost, "Personnel": personnel_cost}

def calculate_TCO_cloud_new(params, years, util):
    instance_count = int(np.ceil(params["instance_count"] * (util / 100)))
    hourly_instance_cost = params["hourly_instance_cost"]
    storage_requirement = params["storage_requirement"]
    monthly_storage_cost = params["monthly_storage_cost"]
    personnel_setup_cost = params["personnel_setup_cost"]
    monthly_personnel_cost = params["monthly_personnel_cost"]
    monthly_network_transfer_cost = params["monthly_network_transfer_cost"]

    ###################################
    # Subcost calculation
    ###################################
    cloud_instance_cost = instance_count * hourly_instance_cost * 24 * 365 * years
    cloud_storage_cost = storage_requirement * monthly_storage_cost * years * 12
    personnel_cost = personnel_setup_cost + monthly_personnel_cost * years * 12
    network_cost = monthly_network_transfer_cost * years * 12
    
    return {"Instance": cloud_instance_cost, "Storage": cloud_storage_cost, "Network": network_cost, "Personnel": personnel_cost}

def calculate_TCO_onpremise_new(params, years):
    node_count = params["node_count"]
    cost_per_node = params["cost_per_node"]
    rack_infra_cost = params["rack_infra_cost"]
    power_infra_cost = params["power_infra_cost"]
    monthly_data_center_cost = params["monthly_data_center_cost"]
    monthly_power = params["monthly_power"]
    power_cost = params["power_cost"]
    personnel_setup_cost = params["personnel_setup_cost"]
    monthly_personnel_cost = params["monthly_personnel_cost"]
    
    ###################################
    # Subcost calculation
    ###################################
    server_cost = node_count * cost_per_node
    server_cost = node_count * cost_per_node
    power_consumption_cost = monthly_power * power_cost * years * 12
    data_center_space_cost = monthly_data_center_cost * years * 12
    personnel_cost = personnel_setup_cost + monthly_personnel_cost * years * 12
    
    return {"Server": server_cost, "Rack Infra": rack_infra_cost, "Power Infra": power_infra_cost, 
            "Power Consumption": power_consumption_cost, "Data Center": data_center_space_cost, "Personnel": personnel_cost}

def display_TCO(params, num_servers, years):
    TCO_data = dict()
    TCO_sums = dict()
    
    sys_type = params["sys_type"]
    sys_name = params["sys_name"]
    
    for i in range(1, years + 1):
        if sys_type == "Cloud":
            TCO = calculate_TCO_cloud(params, num_servers, i)
        else:
            TCO = calculate_TCO_onpremise(params, num_servers, i)
        
        TCO_data[i] = TCO
        TCO_sums[i] = sum(TCO.values())
        
    display(HTML("<hr>"))
    
    table_data = [["Cost Type", "Cost in USD"]]
    TCO = TCO_data[years]
    for key in TCO.keys():
        table_data.append([key, "{:.2f}".format(TCO[key])])
    table_data.append(["<b>Total</b>", "{:.2f}".format(TCO_sums[years])])
    
    print_title("{}-Year TCO for {} ({})".format(years, sys_name, sys_type), "h3")
    print_table(table_data)
    
    ##############################
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, print_grid=False)
    
    for key in TCO.keys():
        trace = go.Bar(x=["{} ({})".format(sys_name, sys_type)], y=[TCO[key]], name=key)
        fig.append_trace(trace, 1, 1)

    trace = go.Scatter(x=["Year {0}".format(key) for key in TCO_sums.keys()], y=list(TCO_sums.values()), mode="lines+markers", name="TCO")
    fig.append_trace(trace, 1, 2)

    #fig["layout"]["title"] = "{0} Year TCO with {1}% Cluster Node Usage".format(years, int(usage))
    fig["layout"]["legend"].update(font=dict(size=12), traceorder="normal")
    fig["layout"]["barmode"] = "stack"
    fig["layout"]["yaxis"].update(title="USD")
    iplot(fig, show_link=False)
    
def display_compare_TCO(params_list, years, util, df, record_id_list, df_xy_list, **nums_servers):
    colors_dict = {"Instance": COLORS_10[0], "Storage": COLORS_10[1], "Network": COLORS_10[2], "Server": COLORS_10[3], 
                   "Rack Infra": COLORS_10[4], "Power Infra": COLORS_10[5], "Power Consumption": COLORS_10[6], 
                   "Data Center": COLORS_10[7], "Personnel": COLORS_10[8]}
    
    if len(params_list) == 0:
        return 
    
    num_systems = len(params_list)
    
    display(HTML("<hr>"))
    
    ###########################################################################################

    fig = make_subplots(rows=1, cols=num_systems, shared_yaxes=True, print_grid=False)
    fig2 = make_subplots(rows=1, cols=1, print_grid=False)
    
    table_data = []
    annotations = []
    
    for j in range(num_systems):
        params = params_list[j]
        
        org_name = params["org_name"]
        sys_name = params["sys_name"]
        num_servers = nums_servers["num_servers_{}".format(j)]
        sys_type = params["sys_type"]
        
        TCO_data = dict()
        TCO_sums = dict()

        for i in range(0, years + 1):
            if sys_type == "Cloud":
                TCO = calculate_TCO_cloud(params, num_servers, i, util)
            else:
                TCO = calculate_TCO_onpremise(params, num_servers, i)

            TCO_data[i] = TCO
            TCO_sums[i] = sum(TCO.values())
        
        table_data.append([TCO_data[years], TCO_sums[years]])
        
        x = "{} {}<br>{}".format(org_name, sys_name, sys_type)
        for key in TCO.keys():
            trace = go.Bar(x=[x], y=[TCO[key]], 
                           name=key, marker=dict(color=colors_dict[key]))
            fig.append_trace(trace, 1, j+1)

        annotations.append(dict(x=x, y=TCO_sums[years]+1000000, xref="x{}".format(j+1), yref="y", 
                                text="$ {:,.0f}".format(TCO_sums[years]), showarrow=False))
        
        trace = go.Scatter(x=["Year {0}".format(key) for key in TCO_sums.keys()], y=list(TCO_sums.values()), 
                           mode="lines+markers", name=x)
        fig2.append_trace(trace, 1, 1)
        
    fig["layout"]["height"] = 400
    fig["layout"]["title"] = "{}-Year TCO".format(years)
    fig["layout"]["barmode"] = "stack"
    fig["layout"]["yaxis"].update(title="USD")
    fig["layout"]["showlegend"] = False
    fig["layout"]["annotations"] = annotations
    fig["layout"]["margin"] = go.layout.Margin(b=35, pad=1)
    iplot(fig, show_link=False)
    
    trace_list = []
    for k in range(num_systems):
        
        tco = table_data[k][0]
        tco_sum = table_data[k][1]
        org_name = params_list[k]["org_name"]
        sys_name = params_list[k]["sys_name"]
        sys_type = params_list[k]["sys_type"]

        keys = list(tco.keys())
        data = ["{:,.0f}".format(tco[key]) for key in keys]
        
        keys.append("<b>Total</b>")
        data.append("<b>{:,.0f}</b>".format(tco_sum))
        
        width = float(1) / num_systems
        trace = go.Table(domain=dict(x=[k*width if k==0 else k*width+0.03, (k+1)*width], y=[0, 1]), 
                         header=dict(values=["<b>Type</b>", "<b>Cost ($)</b>"], align=['center','center']), 
                         cells=dict(values=[keys, data], align=['left','right']))
        trace_list.append(trace)
        
    layout = go.Layout(height=300, margin=go.layout.Margin(l=80, r=75, b=5, t=5, pad=1))
    fig3 = dict(data=trace_list, layout=layout)
    iplot(fig3, show_link=False)
    
    fig2["layout"]["height"] = 300
    fig2["layout"]["yaxis"].update(title="USD")
    fig2["layout"]["showlegend"] = True
    fig2["layout"]["margin"] = go.layout.Margin(b=1, t=5, pad=1)
    iplot(fig2, show_link=False)

    
    
    
def display_compare_TCO_new(years, util, df, record_id_list, df_xy_list, **params_all):
    if len(params_all) == 0:
        return 
    
    colors_dict = {"Instance": COLORS_10[0], "Storage": COLORS_10[1], "Network": COLORS_10[2], "Server": COLORS_10[3], 
                   "Rack Infra": COLORS_10[4], "Power Infra": COLORS_10[5], "Power Consumption": COLORS_10[6], 
                   "Data Center": COLORS_10[7], "Personnel": COLORS_10[8]}
    
    ###########################################################################################
    fig_list = []
    lr_dict = {}
    
    for record_id, df_xy in zip(record_id_list, df_xy_list):
        sys_type = df["Type"][record_id]
        org_name = df["Org"][record_id]
        sys_name = df["System"][record_id]
        x_unit = df["X"][record_id]
        y_unit = df["Y"][record_id]
        degree = params_all[str(record_id)]["degree"]
        
        if len(df_xy) == 1:
            df_xy = df_xy.append({"x": 0, "y":0}, ignore_index=True)
            
        x_vals = np.array(df_xy.x)
        y_vals = np.array(df_xy.y)
        
        max_x = max(x_vals) * 3
        
        X = x_vals
        y = y_vals

        X_reshape = X.reshape(-1, 1)
        poly = PolynomialFeatures(degree=degree)
        X_reshape_poly = poly.fit_transform(X_reshape)

        lr = LinearRegression()
        lr.fit(X_reshape_poly, y)
        lr_dict[str(record_id)] = lr

        X_new = np.arange(1, int(max_x)+1)
        X_new_reshape = X_new.reshape(-1, 1)
        X_new_reshape_poly = poly.fit_transform(X_new_reshape)
        y_new_pred = lr.predict(X_new_reshape_poly)

        trace0 = go.Scatter(x=X_new, y=y_new_pred, mode="lines", name=x_unit)
        trace1 = go.Scatter(x=X, y=y, mode="markers", marker=dict(size=15))
        
        layout = go.Layout(title="Performance Scaling for {} {} {} (ID {})".format(org_name, sys_name, sys_type, record_id),
                           height=300, showlegend=False, hovermode="closest", 
                           xaxis=dict(title="Number of {}".format("Instances" if sys_type == "Cloud" else "Nodes"), showspikes=True),
                           yaxis=dict(title=y_unit, showspikes=True)
                          )
        
        fig4 = dict(data=[trace0, trace1], layout=layout)
        fig_list.append(fig4)
    
    ###########################################################################################

    num_systems = len(params_all)
    fig = make_subplots(rows=1, cols=num_systems, shared_yaxes=True, print_grid=False)
    
    table_data = []
    annotations = []
    trace_list_fig2 = []
    
    for j in range(num_systems):
        record_id = list(params_all.keys())[j]
        params = params_all[record_id]
        
        org_name = params["org_name"]
        sys_name = params["sys_name"]
        sys_type = params["sys_type"]
        num_servers = params["instance_count"] if sys_type == "Cloud" else params["node_count"]
        degree = params["degree"]
        
        ########################################################
        X_new = np.array([num_servers])
        X_new_reshape = X_new.reshape(-1, 1)
        poly = PolynomialFeatures(degree=degree)
        X_new_reshape_poly = poly.fit_transform(X_new_reshape)
        
        lr = lr_dict[record_id]
        y_new_pred = lr.predict(X_new_reshape_poly)
        rmax = y_new_pred[0]
        ########################################################

        TCO_data = dict()
        TCO_sums = dict()

        for i in range(0, years + 1):
            if sys_type == "Cloud":
                TCO = calculate_TCO_cloud_new(params, i, util)
            else:
                TCO = calculate_TCO_onpremise_new(params, i)

            TCO_data[i] = TCO
            TCO_sums[i] = sum(TCO.values())
        
        table_data.append([TCO_data[years], TCO_sums[years]])
        
        x = "{} {}<br>{}".format(org_name, sys_name, sys_type)
        for key in TCO.keys():
            trace = go.Bar(x=[x], y=[TCO[key]], 
                           name=key, marker=dict(color=colors_dict[key]))
            fig.append_trace(trace, 1, j+1)

        annotations.append(dict(x=x, y=TCO_sums[years]+2500000, xref="x{}".format(j+1), yref="y", 
                                text="$ {:,.0f}".format(TCO_sums[years]), showarrow=False))
        annotations.append(dict(x=x, y=TCO_sums[years]+1000000, xref="x{}".format(j+1), yref="y", 
                                text="(Rmax {:.2f} TFLOPS)".format(rmax), showarrow=False))
        
        trace = go.Scatter(x=["Year {0}".format(key) for key in TCO_sums.keys()], y=list(TCO_sums.values()), 
                           mode="lines+markers", name=x)
        trace_list_fig2.append(trace)
        
    fig["layout"]["height"] = 400
    fig["layout"]["title"] = "{}-Year TCO".format(years)
    fig["layout"]["barmode"] = "stack"
    fig["layout"]["yaxis"].update(title="USD")
    fig["layout"]["showlegend"] = False
    fig["layout"]["annotations"] = annotations
    fig["layout"]["margin"] = go.layout.Margin(b=35, pad=1)
    iplot(fig, show_link=False)
    
    ###########################################################################################
    
    trace_list_fig3 = []
    for k in range(num_systems):
        record_id = list(params_all.keys())[k]
        params = params_all[record_id]
        
        tco = table_data[k][0]
        tco_sum = table_data[k][1]
        org_name = params_all[record_id]["org_name"]
        sys_name = params_all[record_id]["sys_name"]
        sys_type = params_all[record_id]["sys_type"]

        keys = list(tco.keys())
        data = ["{:,.0f}".format(tco[key]) for key in keys]
        
        keys.append("<b>Total</b>")
        data.append("<b>{:,.0f}</b>".format(tco_sum))
        
        width = float(1) / num_systems
        trace = go.Table(domain=dict(x=[k*width if k==0 else k*width+0.03, (k+1)*width], y=[0, 1]), 
                         header=dict(values=["<b>Type</b>", "<b>Cost ($)</b>"], align=['center','center']), 
                         cells=dict(values=[keys, data], align=['left','right']))
        trace_list_fig3.append(trace)
        
    layout = go.Layout(height=300, margin=go.layout.Margin(l=80, r=75, b=5, t=5, pad=1))
    fig3 = dict(data=trace_list_fig3, layout=layout)
    iplot(fig3, show_link=False)
    
    layout = go.Layout(height=300, showlegend=True, margin=go.layout.Margin(t=5, pad=1),
                       yaxis=dict(title="USD")
                      )
    fig2 = dict(data=trace_list_fig2, layout=layout)
    iplot(fig2, show_link=False)
    
    ###########################################################################################
    
    for fig in fig_list:
        iplot(fig, show_link=False)
        
    display(HTML("<hr>"))
    
def analyze_tco(q, df, user_id):
    if len(q.get_selected_rows()) == 0:
        q.change_selection([1, 2, 3])

    record_id_list = sorted(q.get_selected_df().index)
    df_xy_list = []

    for record_id in record_id_list:
        user_id = df["UserID"][record_id]
        ts = df["UploadTime"][record_id]
        org_name = df["Org"][record_id]
        sys_name = df["System"][record_id]
        sys_type = df["Type"][record_id]
        x = df["X"][record_id]
        y = df["Y"][record_id]

        records = myDB.get_selected_benchmark_records(user_id, ts)
        df_xy = pd.DataFrame(records, columns=["x", "y"])
        df_xy_list.append(df_xy)

    ###############################################################    

    tco_widgets = {}

    style = {"description_width": '300px'}
    layout = Layout(width="45%")

    for record_id in record_id_list:
        user_id = df["UserID"][record_id]
        ts = df["UploadTime"][record_id]
        org_name = df["Org"][record_id]
        sys_name = df["System"][record_id]
        sys_type = df["Type"][record_id]

        w_title = widgets.HTML(
            value = "<b>{} {} {}</b>".format(org_name, sys_name, sys_type),
            description = "ID {}: ".format(record_id),
            style = {"description_width": '50px'},
            layout = layout
        )
        tco_widgets["{} title".format(record_id)] = w_title

        cursor = myDB.get_tco_questions(sys_type)
        cursor2 = myDB.get_tco_data(sys_type, user_id, ts)

        tco_values = list(cursor2)[0]

        for item, tco_value in zip(list(cursor), tco_values[3:]):
            question = item[0]
            variable_name = item[1]
            unit = item[2]

            w = widgets.FloatText(
                description = "{} {}".format(question, "" if unit == "" else "({})".format(unit)),
                value = tco_value,
                layout = layout,
                style = style
            )

            tco_widgets["{} {}".format(record_id, variable_name)] = w

        w_degree = widgets.IntSlider(
            value = 2,
            min = 1,
            max = 3,
            step = 1,
            description = "Degree of Polynomial for Performance Scaling".format(record_id),
            layout = layout,
            style = style
        )

        tco_widgets["{} degree".format(record_id)] = w_degree

    w_br = widgets.HTML(
        value = "<br>",
        description = " "
    )
    tco_widgets["br"] = w_br

    style = {'description_width': '150px'}
    layout = Layout(width="70%")

    w_years = widgets.IntSlider(
        description = "Life Span (Years)",
        min = 0,
        max = MAX_YEARS,
        step = 1,
        value = 5,
        style = style,
        layout = layout
    )
    tco_widgets["years"] = w_years

    w_util = widgets.IntSlider(
        description = "System Utilization (%)",
        min = 0,
        max = 100,
        step = 5,
        value = 100,
        style = style, 
        layout = layout
    )
    tco_widgets["util"] = w_util

    def view(**tco_widgets):
        params_all = {}
        years = tco_widgets["years"]
        util = tco_widgets["util"]

        for key in tco_widgets:
            if key in ["years", "util", "br"]:
                continue

            record_id, variable_name = key.split()

            assert int(record_id) in record_id_list

            if record_id not in params_all:
                params_all[record_id] = dict()

            params_all[str(record_id)][variable_name] = tco_widgets[key]

        for record_id in record_id_list:
            params_all[str(record_id)]["org_name"] = df["Org"][record_id] 
            params_all[str(record_id)]["sys_name"] = df["System"][record_id]
            params_all[str(record_id)]["sys_type"] = df["Type"][record_id] 

        display_compare_TCO_new(years, util, df, record_id_list, df_xy_list, **params_all)

    i = interactive(view, **tco_widgets)
    display(i)
    
def save_excel(user_id, excel_file, sheets, is_public):
    if is_public == "Keep it public":
        is_public = 1
    else:
        is_public = 0
        
    query_list = []
        
    for sheet in sheets.split(","):
        sheet = sheet.strip()
        sheet = sheet.strip('""')
        
        try:
            df_ = pd.read_excel(excel_file, sheet_name=sheet, header=None, use_cols=True)
        except FileNotFoundError:
            print("ERROR: There is no file named {} on your Jupyter Notebook home directory.".format(excel_file))
            return 
        except XLRDError:
            print("ERROR: There is no spreadsheet named {} in your Excel file.".format(sheet))
            return
        
        df_.columns = columns=["A", "B", "C", "D", "E", "F", "G"]
        df_.index = range(1, len(df_)+1)
        df_ = df_.drop(["A"], axis=1)

        ts = str(dt.now())
        
        params = {"user_id": user_id, "ts": ts, "org_name": df_.C[7], "sys_name": df_.C[8], 
                  "sys_desc": df_.C[9], "sys_type": df_.C[10], "deploy_date": str(df_.C[11])[:12], "is_public": is_public}

        if not pd.isna(df_.C[14]):
            params["is_benchmark"] = "Yes"
            params["measure_name"] = df_.C[14]
            params["x_name"] = df_.C[15]
            params["y_name"] = df_.C[16]
            params["benchmark_date"] = str(df_.C[17])[:12]
            
            x_list, y_list = [], []
            for i in range(20, 20 + 3 * 5, 3):
                x_val = df_.C[i]
                y_val = df_.C[i + 1]

                if (not pd.isna(x_val)) and (not pd.isna(y_val)):
                    x_list.append(x_val)
                    y_list.append(y_val)

            params["x_list"] = x_list
            params["y_list"] = y_list
        else:
            params["is_benchmark"] = "No"

        if not pd.isna(df_.C[36]):
            params["is_tco"] = "Yes"
            params["node_count"] = df_.C[36]
            params["cost_per_node"] = df_.C[37]
            params["rack_infra_cost"] = df_.C[38]
            params["power_infra_cost"] = df_.C[39]
            params["monthly_data_center_cost"] = df_.C[40]
            params["monthly_power"] = df_.C[41]
            params["power_cost"] = df_.C[42]
            params["personnel_setup_cost"] = df_.C[43]
            params["monthly_personnel_cost"] = df_.C[44]
        elif not pd.isna(df_.F[36]):
            params["is_tco"] = "Yes"
            params["instance_count"] = df_.F[36]
            params["hourly_instance_cost"] = df_.F[37]
            params["storage_requirement"] = df_.F[38]
            params["monthly_storage_cost"] = df_.F[39]
            params["monthly_network_transfer_cost"] = df_.F[40]
            params["personnel_setup_cost"] = df_.F[41]
            params["monthly_personnel_cost"] = df_.F[42]
        else:
            params["is_tco"] = "No"
            
        myDB.save_data(**params)
            
    print("Data saved at {}.".format(ts))