from data_utils.parse_args import args
import numpy as np
import torch
import data.tasks_NY.tasks_crime, data.tasks_NY.tasks_chk, data.tasks_NY.tasks_serviceCall
import data.tasks_Chi.tasks_crime, data.tasks_Chi.tasks_chk, data.tasks_Chi.tasks_serviceCall
import data.tasks_SF.tasks_crime, data.tasks_SF.tasks_chk, data.tasks_SF.tasks_serviceCall


def load_data():
    data_path = args.data_path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    landUse_feature = np.load(data_path + args.landUse_dist)
    landUse_feature = landUse_feature[np.newaxis]
    landUse_feature = torch.Tensor(landUse_feature).to(device)

    POI_feature = np.load(data_path + args.POI_dist)
    POI_feature = POI_feature[np.newaxis]
    POI_feature = torch.Tensor(POI_feature).to(device)

    mob_feature = np.load(data_path + args.mobility_dist)
    mob_feature = mob_feature[np.newaxis]
    mob_feature = torch.Tensor(mob_feature).to(device)

    mob_adj = np.load(data_path + args.mobility_adj)
    mob_adj = mob_adj/np.mean(mob_adj)
    mob_adj = torch.Tensor(mob_adj).to(device)

    poi_sim = np.load(data_path + args.POI_simi)
    poi_sim = torch.Tensor(poi_sim).to(device)

    land_sim = np.load(data_path + args.landUse_simi)
    land_sim = torch.Tensor(land_sim).to(device)

    features = [POI_feature, landUse_feature, mob_feature]

    return features, mob_adj, poi_sim, land_sim



def test_model(city, task,emb):
    if task == "checkIn":
        if city == "NY":
            print('>>>>>>>>>>>>>>>>>   Check-In in New York City')
            mae, rmse, r2 = data.tasks_NY.tasks_chk.do_tasks(emb)
        elif city == "Chi":
            print('>>>>>>>>>>>>>>>>>   Check-In in Chicago')
            mae, rmse, r2 = data.tasks_Chi.tasks_chk.do_tasks(emb)
        elif city == "SF":
            print('>>>>>>>>>>>>>>>>>   Check-In in San Francisco')
            mae, rmse, r2 = data.tasks_SF.tasks_chk.do_tasks(emb)
    elif task == "crime":
        if city == "NY":
            print('>>>>>>>>>>>>>>>>>   Crime in New York City')
            mae, rmse, r2 = data.tasks_NY.tasks_crime.do_tasks(emb)
        elif city == "Chi":
            print('>>>>>>>>>>>>>>>>>   Crime in Chicago')
            mae, rmse, r2 = data.tasks_Chi.tasks_crime.do_tasks(emb)
        elif city == "SF":
            print('>>>>>>>>>>>>>>>>>   Crime in San Francisco')
            mae, rmse, r2 = data.tasks_SF.tasks_crime.do_tasks(emb)
    elif task == "serviceCall":
        if city == "NY":
            print('>>>>>>>>>>>>>>>>>   Service Calls in New York City')
            mae, rmse, r2 = data.tasks_NY.tasks_serviceCall.do_tasks(emb)
        elif city == "Chi":
            print('>>>>>>>>>>>>>>>>>   Service Calls in Chicago')
            mae, rmse, r2 = data.tasks_Chi.tasks_serviceCall.do_tasks(emb)
        elif city == "SF":
            print('>>>>>>>>>>>>>>>>>   Service Calls in San Francisco')
            mae, rmse, r2 = data.tasks_SF.tasks_serviceCall.do_tasks(emb)

    return mae, rmse, r2
