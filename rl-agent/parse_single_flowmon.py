import argparse
import json
import re
import xml.etree.ElementTree as ET


def parse_ns3_time_seconds(value):
    if value is None:
        return 0.0

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return 0.0

    match = re.fullmatch(r"([+-]?[0-9]*\.?[0-9]+(?:e[+-]?\d+)?)\s*(ns|us|ms|s)?", text, re.IGNORECASE)
    if not match:
        raise ValueError(f"Unsupported ns-3 time value: {value}")

    magnitude = float(match.group(1))
    unit = (match.group(2) or "s").lower()
    scale = {
        "ns": 1e-9,
        "us": 1e-6,
        "ms": 1e-3,
        "s": 1.0,
    }[unit]
    return magnitude * scale


def parse_flow_int(flow_elem):
    return int(flow_elem.get("flowId", 0))


def build_classifier_map(root):
    classifier = {}
    for flow_elem in root.findall("./Ipv4FlowClassifier/Flow"):
        flow_id = parse_flow_int(flow_elem)
        classifier[flow_id] = {
            "source_address": flow_elem.get("sourceAddress"),
            "destination_address": flow_elem.get("destinationAddress"),
            "protocol": flow_elem.get("protocol"),
            "source_port": int(flow_elem.get("sourcePort", 0)),
            "destination_port": int(flow_elem.get("destinationPort", 0)),
        }
    return classifier


def get_top_level_flows(root):
    return root.findall("./FlowStats/Flow")


def get_flow_classifier(flow_elem, classifier_map):
    flow_id = parse_flow_int(flow_elem)
    return classifier_map.get(flow_id, {})


def get_flow_rx_bytes(flow_elem):
    return int(flow_elem.get("rxBytes", 0))


def get_flow_tx_bytes(flow_elem):
    return int(flow_elem.get("txBytes", 0))


def choose_primary_flow(flow_elems, classifier_map, destination_port=None):
    if not flow_elems:
        raise ValueError("No top-level FlowStats/Flow entries found")

    enriched_flows = []
    for flow_elem in flow_elems:
        classifier = get_flow_classifier(flow_elem, classifier_map)
        rx_bytes = get_flow_rx_bytes(flow_elem)
        tx_bytes = get_flow_tx_bytes(flow_elem)
        enriched_flows.append((flow_elem, classifier, rx_bytes, tx_bytes))

    if destination_port is not None:
        matching = [
            item for item in enriched_flows if item[1].get("destination_port") == destination_port
        ]
        if matching:
            return max(matching, key=lambda item: (item[2], item[3]))[0]

    tcp_flows = [item for item in enriched_flows if item[1].get("protocol") in {None, "6"}]
    if tcp_flows:
        return max(tcp_flows, key=lambda item: (item[2], item[3]))[0]

    return max(enriched_flows, key=lambda item: (item[2], item[3]))[0]


def compute_flow_duration_seconds(flow_elem, root):
    start_time = parse_ns3_time_seconds(flow_elem.get("timeFirstTxPacket"))
    end_time = parse_ns3_time_seconds(flow_elem.get("timeLastRxPacket"))

    if end_time > start_time:
        return end_time - start_time

    fallback_end = 0.0
    for candidate in get_top_level_flows(root):
        fallback_end = max(
            fallback_end,
            parse_ns3_time_seconds(candidate.get("timeLastRxPacket")),
            parse_ns3_time_seconds(candidate.get("timeLastTxPacket")),
        )

    if fallback_end > 0.0:
        return fallback_end

    return 0.0


def compute_flow_throughput_mbps(flow_elem, root):
    rx_bytes = get_flow_rx_bytes(flow_elem)
    flow_duration_sec = compute_flow_duration_seconds(flow_elem, root)
    if flow_duration_sec <= 0.0:
        return 0.0
    return (rx_bytes * 8.0) / (flow_duration_sec * 1e6)


def select_fairness_flows(flow_elems, classifier_map, primary_flow, destination_port=None):
    if destination_port is not None:
        fairness_flows = [
            flow_elem
            for flow_elem in flow_elems
            if get_flow_classifier(flow_elem, classifier_map).get("destination_port") == destination_port
        ]
        if fairness_flows:
            return fairness_flows

    primary_destination_port = get_flow_classifier(primary_flow, classifier_map).get("destination_port")
    if primary_destination_port is not None:
        fairness_flows = [
            flow_elem
            for flow_elem in flow_elems
            if get_flow_classifier(flow_elem, classifier_map).get("destination_port") == primary_destination_port
        ]
        if fairness_flows:
            return fairness_flows

    return [primary_flow]


def compute_jain_fairness(flow_elems, root):
    throughputs = [compute_flow_throughput_mbps(flow_elem, root) for flow_elem in flow_elems]
    throughputs = [throughput for throughput in throughputs if throughput > 0.0]

    if not throughputs:
        return 0.0, 0

    sum_tp = sum(throughputs)
    sum_sq = sum(throughput ** 2 for throughput in throughputs)
    if sum_sq <= 0.0:
        return 0.0, len(throughputs)

    n_flows = len(throughputs)
    fairness = (sum_tp ** 2) / (n_flows * sum_sq)
    return fairness, n_flows


def parse_flowmon_xml(xml_path, destination_port=None):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    flow_elems = get_top_level_flows(root)
    classifier_map = build_classifier_map(root)
    primary_flow = choose_primary_flow(flow_elems, classifier_map, destination_port=destination_port)
    fairness_flows = select_fairness_flows(
        flow_elems,
        classifier_map,
        primary_flow,
        destination_port=destination_port,
    )

    flow_id = parse_flow_int(primary_flow)
    classifier = classifier_map.get(flow_id, {})
    tx_packets = int(primary_flow.get("txPackets", 0))
    rx_packets = int(primary_flow.get("rxPackets", 0))
    lost_packets = int(primary_flow.get("lostPackets", 0))
    rx_bytes = get_flow_rx_bytes(primary_flow)
    delay_sum_sec = parse_ns3_time_seconds(primary_flow.get("delaySum"))
    flow_duration_sec = compute_flow_duration_seconds(primary_flow, root)
    jain_fairness, n_flows = compute_jain_fairness(fairness_flows, root)

    throughput_mbps = compute_flow_throughput_mbps(primary_flow, root)
    avg_delay_ms = (delay_sum_sec * 1e3 / rx_packets) if rx_packets > 0 else 0.0
    loss_rate = (lost_packets / tx_packets) if tx_packets > 0 else 0.0

    return {
        "flow_id": flow_id,
        "source_address": classifier.get("source_address"),
        "destination_address": classifier.get("destination_address"),
        "source_port": classifier.get("source_port"),
        "destination_port": classifier.get("destination_port"),
        "throughput_mbps": throughput_mbps,
        "avg_delay_ms": avg_delay_ms,
        "loss_rate": loss_rate,
        "rx_packets": rx_packets,
        "tx_packets": tx_packets,
        "lost_packets": lost_packets,
        "rx_bytes": rx_bytes,
        "flow_duration_sec": flow_duration_sec,
        "jain_fairness": jain_fairness,
        "n_flows": n_flows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("xml_file")
    parser.add_argument("--destination-port", type=int, default=None)
    args = parser.parse_args()

    metrics = parse_flowmon_xml(args.xml_file, destination_port=args.destination_port)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()