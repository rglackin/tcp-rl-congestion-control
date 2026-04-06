#include "ns3/core-module.h"
#include "ns3/opengym-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/flow-monitor-helper.h"
#include "ns3/tcp-rl.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("OpenGym");

// Global stats for RL environment
static double   g_envStepTime = 0.1;
static uint64_t g_total_steps = 10;
static uint64_t g_currRxBytes = 0;
static uint32_t g_stepCount = 0;
static uint32_t g_rewardId = 1;
static std::string g_mode = "rl";
static double g_bottleneckRateMbps = 10.0;
static std::string g_flowmonXml = "";
static Ptr<OnOffApplication> g_clientApp = nullptr;
static Ptr<PacketSink> g_sinkApp = nullptr;
static Ptr<FlowMonitor> g_flowMonitor = nullptr;

static double
ComputeThroughputMbps (uint64_t currRxBytes, uint64_t& prevRxBytes)
{
  uint64_t deltaBytes = currRxBytes - prevRxBytes;
  prevRxBytes = currRxBytes;

  if (g_envStepTime <= 0.0)
    {
      return 0.0;
    }

  return (deltaBytes * 8.0) / (g_envStepTime * 1e6);
}

static double
ComputeNormCwnd (double rttSec)
{
  double bdpBytes = g_bottleneckRateMbps * 1e6 * rttSec / 8.0;
  Ptr<RlTcpController> controller = TcpRl::GetController ();
  double lastCwndBytes = controller ? controller->GetLastCwndBytes () : 0.0;
  return (bdpBytes > 0.0) ? (lastCwndBytes / bdpBytes) : 0.0;
}

static double
ComputeUtilization (double throughputMbps)
{
  if (g_bottleneckRateMbps <= 0.0)
    {
      return 0.0;
    }

  return std::min(throughputMbps / g_bottleneckRateMbps, 1.0);
}

static Ptr<OpenGymDataContainer>
BuildObservation (double throughputMbps, double normCwnd, double util, double rttRatio)
{
  std::vector<uint32_t> shape = {4,};
  Ptr<OpenGymBoxContainer<float>> box =
      CreateObject<OpenGymBoxContainer<float>>(shape);

  box->AddValue((float)throughputMbps);
  box->AddValue((float)normCwnd);
  box->AddValue((float)util);
  box->AddValue((float)rttRatio);

  return box;
}

/*
Define observation space
*/
Ptr<OpenGymSpace> MyGetObservationSpace(void)
{
  // 4 features: [throughputMbps, normCwnd, utilization, rttRatio]
  float low = 0.0;
  float high = 1.0;

  std::vector<uint32_t> shape = {4,};
  std::string dtype = TypeNameGet<float>();

  Ptr<OpenGymBoxSpace> space =
      CreateObject<OpenGymBoxSpace>(low, high, shape, dtype);

  NS_LOG_DEBUG("MyGetObservationSpace: " << space);
  return space;
}

/*
Define action space
*/
Ptr<OpenGymSpace> MyGetActionSpace(void)
{
  uint32_t nActions = 5;

  Ptr<OpenGymDiscreteSpace> space = CreateObject<OpenGymDiscreteSpace> (nActions);
  NS_LOG_DEBUG ("MyGetActionSpace: " << space);
  return space;
}

/*
Define game over condition
*/
bool MyGetGameOver(void)
{
  NS_LOG_DEBUG("GameOver: step=" << g_stepCount << "/" << g_total_steps 
               << " time=" << Simulator::Now().GetSeconds());
  bool isGameOver = (g_stepCount >= g_total_steps);
  
  NS_LOG_DEBUG ("MyGetGameOver: " << isGameOver);
  return isGameOver;
}

/*
Collect observations
*/
Ptr<OpenGymDataContainer> MyGetObservation(void)
{
  g_currRxBytes = g_sinkApp->GetTotalRx();
  static uint64_t prevRxBytes = 0;
  double throughputMbps = ComputeThroughputMbps (g_currRxBytes, prevRxBytes);

  Ptr<RlTcpController> controller = TcpRl::GetController ();
  double rttSec = controller ? controller->GetRttSec () : 0.1;
  double rttRatio = controller ? controller->GetRttRatio () : 1.0;

  double normCwnd = ComputeNormCwnd (rttSec);
  double util = ComputeUtilization (throughputMbps);

  NS_LOG_INFO("obs: step=" << g_stepCount
              << " thru=" << throughputMbps
              << " cwndNorm=" << normCwnd
              << " util=" << util
              << " rttRatio=" << rttRatio);
  return BuildObservation (throughputMbps, normCwnd, util, rttRatio);
}

/*
Define reward function
*/
float MyGetReward(void)
{
  uint64_t currRxBytes = g_sinkApp->GetTotalRx();
  static uint64_t prevRxBytes = 0;
  double throughputMbps = ComputeThroughputMbps (currRxBytes, prevRxBytes);
  Ptr<RlTcpController> controller = TcpRl::GetController ();
  double rttSec = controller ? controller->GetRttSec () : 0.1;
  double normCwnd = ComputeNormCwnd (rttSec);
  double util = ComputeUtilization (throughputMbps);
  
  float reward = 0.0;
  switch (g_rewardId) {
    case 1:
      reward = throughputMbps;
      break;
    case 2: {
      const double alpha = 5.0;
      double excess = std::max(0.0, normCwnd - 1.0);
      reward = throughputMbps - alpha * excess;
      break;
    }
    case 3:
      reward = throughputMbps - 0.5 * std::max(0.0, util - 0.8);
      break;
    case 4:
      reward = std::log(throughputMbps + 1e-6);
      break;
    default:
      reward = throughputMbps;
      break;
  }
  
  NS_LOG_INFO("reward: step=" << g_stepCount
              << " rewardId=" << g_rewardId
              << " reward=" << reward);
  return reward;
}

/*
Define extra info. Optional
*/
std::string MyGetExtraInfo(void)
{
  std::string myInfo = "testInfo";
  myInfo += "|123";
  NS_LOG_DEBUG("MyGetExtraInfo: " << myInfo);
  return myInfo;
}


/*
Execute received actions
*/
bool MyExecuteActions(Ptr<OpenGymDataContainer> action)
{
  if (g_mode == "newreno")
    {
      NS_LOG_INFO("action: step=" << g_stepCount
                  << " mode=newreno act=ignored");
      return true;
    }

  Ptr<OpenGymDiscreteContainer> discrete = DynamicCast<OpenGymDiscreteContainer>(action);
  uint32_t actIdx = discrete->GetValue();
  int32_t k = static_cast<int32_t>(actIdx) - 2;

  Ptr<RlTcpController> controller = TcpRl::GetController ();
  if (controller)
    {
      controller->SetAction (k);
    }

  NS_LOG_INFO("action: step=" << g_stepCount
              << " actIdx=" << actIdx
              << " k=" << k);
  return true;
}

void ScheduleNextStateRead(double envStepTime, Ptr<OpenGymInterface> openGym)
{
  NS_LOG_INFO("step: count=" << g_stepCount << " time=" << Simulator::Now().GetSeconds());
  if (g_flowMonitor)
    {
      g_flowMonitor->CheckForLostPackets();
    }
  openGym->NotifyCurrentState();

  g_stepCount++;

  // If the episode is over, stop the simulator and dont reschedule
  if (MyGetGameOver()) {
    NS_LOG_INFO("done: step=" << g_stepCount << " total_steps=" << g_total_steps);
    Simulator::Stop();  // allow Run() to return
    return;
  }

  // Otherwise schedule the next step
  Simulator::Schedule(Seconds(envStepTime), &ScheduleNextStateRead, envStepTime, openGym);
}

int
main (int argc, char *argv[])
{
  // Parameters of the scenario
  uint32_t simSeed = 1;

  double simulationTime = 1; //seconds
  double envStepTime = 0.1; //seconds, ns3gym env step time interval
  
  uint32_t openGymPort = 5555;
  uint32_t testArg = 0;
  uint32_t nFlows = 2;
  std::string bottleneckRate = "10Mbps";
  std::string bottleneckDelay = "50ms";
  std::string accessRate = "100Mbps";
  std::string accessDelay = "1ms";

  CommandLine cmd;
  // required parameters for OpenGym interface
  cmd.AddValue ("openGymPort", "Port number for OpenGym env. Default: 5555", openGymPort);
  cmd.AddValue ("simSeed", "Seed for random generator. Default: 1", simSeed);
  // optional parameters
  cmd.AddValue ("simTime", "Simulation time in seconds. Default: 10s", simulationTime);
  cmd.AddValue ("envStepTime", "Env step time in seconds. Default: 0.1s", envStepTime);
  cmd.AddValue ("testArg", "Extra simulation argument. Default: 0", testArg);
  cmd.AddValue ("nFlows", "Number of TCP flows. Default: 1", nFlows);
  cmd.AddValue ("bottleneckRate", "Bottleneck link data rate. Default: 10Mbps", bottleneckRate);
  cmd.AddValue ("bottleneckDelay", "Bottleneck link delay. Default: 50ms", bottleneckDelay);
  cmd.AddValue ("accessRate", "Access link data rate. Default: 100Mbps", accessRate);
  cmd.AddValue ("accessDelay", "Access link delay. Default: 1ms", accessDelay);
  cmd.AddValue ("rewardId", "Reward function ID (1-4). Default: 1", g_rewardId);
  cmd.AddValue ("mode", "Congestion-control mode: rl or newreno. Default: rl", g_mode);
  cmd.AddValue ("flowmonXml", "Optional FlowMonitor XML output path. Default: disabled", g_flowmonXml);
  cmd.Parse (argc, argv);

  if (g_mode == "newreno")
    {
      Config::SetDefault ("ns3::TcpL4Protocol::SocketType",
                          TypeIdValue (TypeId::LookupByName ("ns3::TcpNewReno")));
      TcpRl::SetController (nullptr);
    }
  else
    {
      if (g_mode != "rl")
        {
          NS_LOG_WARN ("Unknown mode '" << g_mode << "', falling back to rl");
          g_mode = "rl";
        }
      Config::SetDefault ("ns3::TcpL4Protocol::SocketType",
                          TypeIdValue (TcpRl::GetTypeId ()));
      TcpRl::SetController (CreateObject<RlTcpController> ());
    }
  // Set global bottleneck rate in Mbps
  g_bottleneckRateMbps = DataRate(bottleneckRate).GetBitRate() / 1e6;

  NS_LOG_INFO("Ns3Env parameters:");
  NS_LOG_INFO("--simTime: " << simulationTime);
  NS_LOG_INFO("--openGymPort: " << openGymPort);
  NS_LOG_INFO("--envStepTime: " << envStepTime);
  NS_LOG_INFO("--seed: " << simSeed);
  NS_LOG_INFO("--testArg: " << testArg);
  NS_LOG_INFO("--rewardId: " << g_rewardId);
  NS_LOG_INFO("--mode: " << g_mode);
  NS_LOG_INFO("--bottleneckRateMbps: " << g_bottleneckRateMbps);

  RngSeedManager::SetSeed (1);
  RngSeedManager::SetRun (simSeed);

  // Store envStepTime in global variable
  g_envStepTime = envStepTime;

  // Calculate total steps based on simulation time and env step time
  g_total_steps = (uint64_t) (simulationTime / envStepTime);
  NS_LOG_INFO("total_steps=" << g_total_steps);
  // Create dumbbell topology
  NodeContainer leftSenders, routers, rightReceivers;
  leftSenders.Create(nFlows);
  routers.Create(2);
  rightReceivers.Create(nFlows);

  // Install Internet stack on all nodes
  NodeContainer allNodes;
  allNodes.Add(leftSenders);
  allNodes.Add(routers);
  allNodes.Add(rightReceivers);
  InternetStackHelper stack;
  stack.Install(allNodes);

  // Access links
  PointToPointHelper accessLink;
  accessLink.SetDeviceAttribute("DataRate", StringValue(accessRate));
  accessLink.SetChannelAttribute("Delay", StringValue(accessDelay));
  accessLink.SetQueue("ns3::DropTailQueue", "MaxSize", StringValue("100p"));

  // Bottleneck link
  PointToPointHelper bottleneckLink;
  bottleneckLink.SetDeviceAttribute("DataRate", StringValue(bottleneckRate));
  bottleneckLink.SetChannelAttribute("Delay", StringValue(bottleneckDelay));
  bottleneckLink.SetQueue("ns3::DropTailQueue", "MaxSize", StringValue("100p"));

  // Install access links left
  std::vector<NetDeviceContainer> leftDevices(nFlows);
  Ipv4AddressHelper address;
  std::vector<Ipv4InterfaceContainer> leftInterfaces(nFlows);
  for (uint32_t i = 0; i < nFlows; ++i) {
    leftDevices[i] = accessLink.Install(NodeContainer(leftSenders.Get(i), routers.Get(0)));
    std::string base = "10.1." + std::to_string(i+1) + ".0";
    address.SetBase(base.c_str(), "255.255.255.0");
    leftInterfaces[i] = address.Assign(leftDevices[i]);
  }

  // Bottleneck
  NetDeviceContainer bottleneckDevices = bottleneckLink.Install(NodeContainer(routers.Get(0), routers.Get(1)));
  address.SetBase("10.2.0.0", "255.255.255.0");
  Ipv4InterfaceContainer bottleneckInterfaces = address.Assign(bottleneckDevices);

  // Access links right
  std::vector<NetDeviceContainer> rightDevices(nFlows);
  std::vector<Ipv4InterfaceContainer> rightInterfaces(nFlows);
  for (uint32_t i = 0; i < nFlows; ++i) {
    rightDevices[i] = accessLink.Install(NodeContainer(routers.Get(1), rightReceivers.Get(i)));
    std::string base = "10.3." + std::to_string(i+1) + ".0";
    address.SetBase(base.c_str(), "255.255.255.0");
    rightInterfaces[i] = address.Assign(rightDevices[i]);
  }

  // Populate routing tables
  Ipv4GlobalRoutingHelper::PopulateRoutingTables();

  // Install PacketSinks on all receivers
  uint16_t port = 5000;
  for (uint32_t i = 0; i < nFlows; ++i) {
    PacketSinkHelper sinkHelper("ns3::TcpSocketFactory", 
                                 InetSocketAddress(Ipv4Address::GetAny(), port));
    ApplicationContainer sinkApps = sinkHelper.Install(rightReceivers.Get(i));
    sinkApps.Start(Seconds(0.0));
    sinkApps.Stop(Seconds(simulationTime));
    if (i == 0) {
      g_sinkApp = DynamicCast<PacketSink>(sinkApps.Get(0));
    }
  }

  // Install OnOff applications on all senders
  for (uint32_t i = 0; i < nFlows; ++i) {
    OnOffHelper clientHelper("ns3::TcpSocketFactory",
                             InetSocketAddress(rightInterfaces[i].GetAddress(1), port));
    clientHelper.SetAttribute("DataRate", DataRateValue(DataRate("100Mbps")));
    clientHelper.SetAttribute("PacketSize", UintegerValue(1000));
    clientHelper.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
    clientHelper.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
    
    ApplicationContainer clientApps = clientHelper.Install(leftSenders.Get(i));
    clientApps.Start(Seconds(0.1));
    clientApps.Stop(Seconds(simulationTime));
    if (i == 0) {
      g_clientApp = DynamicCast<OnOffApplication>(clientApps.Get(0));
    }
  }

  // Install FlowMonitor
  FlowMonitorHelper flowmonHelper;
  Ptr<FlowMonitor> monitor = flowmonHelper.InstallAll();
  g_flowMonitor = monitor;

  // OpenGym Env
  Ptr<OpenGymInterface> openGym = CreateObject<OpenGymInterface> (openGymPort);
  openGym->SetGetActionSpaceCb( MakeCallback (&MyGetActionSpace) );
  openGym->SetGetObservationSpaceCb( MakeCallback (&MyGetObservationSpace) );
  openGym->SetGetGameOverCb( MakeCallback (&MyGetGameOver) );
  openGym->SetGetObservationCb( MakeCallback (&MyGetObservation) );
  openGym->SetGetRewardCb( MakeCallback (&MyGetReward) );
  openGym->SetGetExtraInfoCb( MakeCallback (&MyGetExtraInfo) );
  openGym->SetExecuteActionsCb( MakeCallback (&MyExecuteActions) );
  Simulator::Schedule (Seconds(0.0), &ScheduleNextStateRead, envStepTime, openGym);

  NS_LOG_INFO ("Simulation start");
  Simulator::Stop (Seconds (simulationTime));
  Simulator::Run ();
  NS_LOG_INFO ("Simulation stop");

  if (!g_flowmonXml.empty()) {
    monitor->CheckForLostPackets();
    monitor->SerializeToXmlFile(g_flowmonXml, true, true);
    NS_LOG_INFO("FlowMonitor XML written: " << g_flowmonXml);
  } else {
    NS_LOG_INFO("FlowMonitor XML disabled (pass --flowmonXml to enable)");
  }

  openGym->NotifySimulationEnd();
  Simulator::Destroy ();

}
