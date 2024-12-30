#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/lte-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/point-to-point-helper.h"
#include "ns3/eps-bearer-tag.h"
#include "ns3/netanim-module.h"
#include "ns3/flow-monitor-helper.h"
#include "ns3/ipv4-flow-classifier.h"
#include <ns3/antenna-module.h>
#include "ns3/trace-source-accessor.h"
#include "ns3/opengym-module.h"

using namespace ns3;

double minPowerDbm = 10.0;
double maxPowerDbm = 40.0;
std::vector<double> cioValues; // Store CIO values for each SBS
double minCio = -10.0; // Minimum CIO value in dB
double maxCio = 10.0; // Maximum CIO value in dB

double CalculateRsrp(Ptr<Node> ueNode, Ptr<Node> gnbNode, double txPowerDbm, double pathLossExponent) {
    Ptr<MobilityModel> ueMobility = ueNode->GetObject<MobilityModel>();
    Ptr<MobilityModel> gnbMobility = gnbNode->GetObject<MobilityModel>();
    double distance = ueMobility->GetDistanceFrom(gnbMobility);
    double txPowerWatts = pow(10.0, txPowerDbm / 10.0) / 1000.0;
    double rsrp = txPowerWatts / pow(distance, pathLossExponent);
    return 10 * log10(rsrp * 1000.0);
}

void PrintEnbMeasurements(NetDeviceContainer& enbDevs) {
    std::cout << "Getting Transmission power for all eNBs..." << std::endl;
    for (uint32_t i = 0; i < enbDevs.GetN(); ++i) {
        Ptr<LteEnbNetDevice> enbDevice = DynamicCast<LteEnbNetDevice>(enbDevs.Get(i));
        Ptr<LteEnbPhy> enbPhy = enbDevice->GetPhy();
        double txPowerDbm = enbPhy->GetTxPower();
        std::cout << "eNB " << i+1 << " transmission power: " << txPowerDbm << " dBm" << std::endl;
    }
}

void PrintConnectedEnbRsrp(NetDeviceContainer& enbDevs, NetDeviceContainer& ueDevs, 
                          double pathLossExponent, Ptr<LteHelper> lteHelper) {
    
    std::cout << "Calculating RSRP for connected eNBs..." << std::endl;
    for (uint32_t i = 0; i < ueDevs.GetN(); ++i) {
        Ptr<LteUeNetDevice> ueDevice = DynamicCast<LteUeNetDevice>(ueDevs.Get(i));
        uint16_t cellId = ueDevice->GetRrc()->GetCellId();
        
        Ptr<LteEnbNetDevice> connectedEnb = nullptr;
        for (uint32_t j = 0; j < enbDevs.GetN(); ++j) {
            Ptr<LteEnbNetDevice> enbDevice = DynamicCast<LteEnbNetDevice>(enbDevs.Get(j));
            if (enbDevice->GetCellId() == cellId) {
                connectedEnb = enbDevice;
                break;
            }
        }
        
        if (connectedEnb != nullptr) {
            Ptr<LteEnbPhy> enbPhy = connectedEnb->GetPhy();
            double txPowerDbm = enbPhy->GetTxPower();
            double rsrp = CalculateRsrp(ueDevs.Get(i)->GetNode(), 
                                        connectedEnb->GetNode(), 
                                        txPowerDbm, 
                                        pathLossExponent);
            std::cout << "RSRP for UE " << i << " to connected eNB (Cell ID " 
                      << cellId << "): " << rsrp << " dBm" << std::endl;
        }
    }
}

void checkUEPosition(NodeContainer& ueNodes){
    for (uint32_t i = 0; i < ueNodes.GetN(); ++i) {
        Ptr<ConstantVelocityMobilityModel> mobility = 
            ueNodes.Get(i)->GetObject<ConstantVelocityMobilityModel>();
        std::cout << "UE " << i << " position: " << mobility->GetPosition() 
                << ", velocity: " << mobility->GetVelocity() << std::endl;
    }
}

// Randomize CIO values for each SBS
void RandomizeCioValues(int numEnb, double minCio, double maxCio) {
    cioValues.clear();
    for (int i = 0; i < numEnb; ++i) {
        double cio = minCio + (maxCio - minCio) * ((double) rand() / RAND_MAX);
        cioValues.push_back(cio);
        std::cout << "eNB " << i << " CIO value: " << cio << " dB" << std::endl;
    }
}

void HandoverDecision(Ptr<LteHelper> lteHelper, Ptr<PointToPointEpcHelper> epcHelper, NodeContainer& ueNodes, 
                     NetDeviceContainer& enbDevs, NetDeviceContainer& ueDevs, double pathLossExponent) {
    NS_ASSERT(lteHelper != nullptr);
    NS_ASSERT(epcHelper != nullptr);
    
    if (enbDevs.GetN() == 0 || ueDevs.GetN() == 0) {
        std::cout << "[WARNING] Empty device containers, skipping handover decision" << std::endl;
        return;
    }

    const double NEIGHBOR_DISTANCE_THRESHOLD = 500.0;
    std::map<uint32_t, uint16_t> currentAttachments;

    // Process each UE
    for (uint32_t i = 0; i < ueDevs.GetN(); ++i) {
        std::cout << "\n[DEBUG] Processing UE " << i << std::endl;

        Ptr<LteUeNetDevice> ueDevice = DynamicCast<LteUeNetDevice>(ueDevs.Get(i));
        if (!ueDevice) {
            std::cout << "[WARNING] Invalid UE device at index " << i << std::endl;
            continue;
        }

        uint16_t currentCellId = ueDevice->GetRrc()->GetCellId();
        if (currentCellId == 0) {
            std::cout << "[WARNING] UE " << i << " is not attached to any eNB" << std::endl;
            continue;
        }

        currentAttachments[i] = currentCellId;
        std::cout << "[DEBUG] UE " << i << " current cell ID: " << currentCellId << std::endl;

        Ptr<LteUeRrc> ueRrc = ueDevice->GetRrc();
        if (!ueRrc || ueRrc->GetState() != LteUeRrc::CONNECTED_NORMALLY) {
            std::cout << "[WARNING] UE " << i << " is not in CONNECTED_NORMALLY state" << std::endl;
            continue;
        }

        uint16_t rnti = ueRrc->GetRnti();
        std::cout << "[DEBUG] UE " << i << " RNTI: " << rnti << std::endl;

        // Find current serving eNB
        Ptr<LteEnbNetDevice> servingEnb = nullptr;
        double servingRsrp = -std::numeric_limits<double>::infinity();
        double servingCio = 0.0;

        std::cout << "[DEBUG] Searching for serving eNB..." << std::endl;
        for (uint32_t j = 0; j < enbDevs.GetN(); ++j) {
            Ptr<LteEnbNetDevice> enbDevice = DynamicCast<LteEnbNetDevice>(enbDevs.Get(j));
            if (!enbDevice || enbDevice->GetCellId() != currentCellId) {
                continue;
            }

            servingEnb = enbDevice;
            Ptr<LteEnbPhy> enbPhy = enbDevice->GetPhy();
            if (!enbPhy) {
                std::cout << "[ERROR] Serving eNB has no PHY layer" << std::endl;
                continue;
            }

            Ptr<MobilityModel> enbMobility = enbDevice->GetNode()->GetObject<MobilityModel>();
            if (enbMobility) {
                double txPowerDbm = enbPhy->GetTxPower();
                servingRsrp = CalculateRsrp(ueDevice->GetNode(), enbDevice->GetNode(), 
                                              txPowerDbm, pathLossExponent);
                servingCio = cioValues[j]; // Get the CIO value for the serving eNB
                std::cout << "[DEBUG] Found serving eNB (CellId: " << currentCellId 
                          << ") with RSRP: " << servingRsrp << " dB and CIO: " << servingCio << " dB" << std::endl;
            }
            break;
        }

        if (!servingEnb) {
            std::cout << "[WARNING] No serving eNB found for UE " << i << std::endl;
            continue;
        }

        // Find best neighbor
        double maxAdjustedRsrp = servingRsrp + servingCio;
        Ptr<LteEnbNetDevice> targetEnbDevice = nullptr;
        uint16_t targetCellId = 0;

        Ptr<MobilityModel> ueMobility = ueDevice->GetNode()->GetObject<MobilityModel>();
        if (!ueMobility) {
            std::cout << "[WARNING] UE has no mobility model at index " << i << std::endl;
            continue;
        }

        Vector uePosition = ueMobility->GetPosition();
        std::cout << "[DEBUG] Checking neighboring eNBs..." << std::endl;

        // Check neighboring eNBs
        for (uint32_t j = 0; j < enbDevs.GetN(); ++j) {
            Ptr<LteEnbNetDevice> enbDevice = DynamicCast<LteEnbNetDevice>(enbDevs.Get(j));
            if (!enbDevice || enbDevice->GetCellId() == currentCellId) {
                continue;
            }

            Ptr<LteEnbPhy> enbPhy = enbDevice->GetPhy();
            if (!enbPhy) {
                std::cout << "[WARNING] Neighboring eNB missing PHY layer" << std::endl;
                continue;
            }

            Ptr<MobilityModel> enbMobility = enbDevice->GetNode()->GetObject<MobilityModel>();
            if (!enbMobility) {
                continue;
            }

            Vector enbPosition = enbMobility->GetPosition();
            double distance = CalculateDistance(uePosition, enbPosition);

            if (distance <= NEIGHBOR_DISTANCE_THRESHOLD) {
                double txPowerDbm = enbPhy->GetTxPower();
                double rsrp = CalculateRsrp(ueDevice->GetNode(), enbDevice->GetNode(), 
                                          txPowerDbm, pathLossExponent);
                double neighborCio = cioValues[j]; // Get the CIO value for the neighbor
                double adjustedRsrp = rsrp + neighborCio;

                std::cout << "[DEBUG] Neighbor eNB (CellId: " << enbDevice->GetCellId() 
                          << ") RSRP: " << rsrp << " dB, CIO: " << neighborCio << " dB"
                          << ", Adjusted RSRP: " << adjustedRsrp << " dB" << std::endl;

                if (adjustedRsrp > maxAdjustedRsrp) {
                    maxAdjustedRsrp = adjustedRsrp;
                    targetEnbDevice = enbDevice;
                    targetCellId = enbDevice->GetCellId();
                }
            }
        }

        // Perform handover if better neighbor found
        if (targetEnbDevice && targetCellId != currentCellId && maxAdjustedRsrp > (servingRsrp + servingCio)) {
            std::cout << "\n[INFO] Triggering handover for UE " << i 
                      << "\n  From CellId: " << currentCellId
                      << " (RSRP + CIO: " << servingRsrp + servingCio << " dB)"
                      << "\n  To CellId: " << targetCellId 
                      << " (RSRP + CIO: " << maxAdjustedRsrp << " dB)"
                      << "\n  Improvement: " << maxAdjustedRsrp - (servingRsrp + servingCio) << " dB" << std::endl;

            lteHelper->HandoverRequest(Seconds(0.1), ueDevice, servingEnb, targetCellId);
            std::cout << "[DEBUG] Handover request sent successfully" << std::endl;
        }
    }

    // Schedule the next handover decision
    const double HANDOVER_CHECK_INTERVAL = 1.0; // Check every 1 second
    Simulator::Schedule(Seconds(HANDOVER_CHECK_INTERVAL), &HandoverDecision, 
                        lteHelper, epcHelper, ueNodes, enbDevs, ueDevs, pathLossExponent);

    std::cout << "\n[INFO] Handover decision completed. Next check scheduled in " 
              << HANDOVER_CHECK_INTERVAL << " seconds.\n" << std::endl;
}

// Custom event to ensure setup is complete before handover in LTE
void WaitForSetupAndTriggerHandover(Ptr<LteHelper> lteHelper, Ptr<PointToPointEpcHelper> epcHelper, NodeContainer& ueNodes, 
                                    NetDeviceContainer& enbDevs, NetDeviceContainer& ueDevs, double pathLossExponent) {
    // Check that setup is complete before scheduling handover decision
    bool setupComplete = true;

    // Ensure all UE devices are attached and have proper RRC connections
    for (uint32_t i = 0; i < ueDevs.GetN(); ++i) {
        Ptr<LteUeNetDevice> ueDevice = DynamicCast<LteUeNetDevice>(ueDevs.Get(i));
        if (!ueDevice || !ueDevice->GetRrc()) {
            setupComplete = false;
            break;
        }
    }

    // If setup is complete, trigger handover decision
    if (setupComplete) {
        // Now that setup is complete, trigger handover decision
        std::cout << "Setup complete. Triggering handover decision..." << std::endl;
        Simulator::Schedule(Seconds(0.0), &HandoverDecision, lteHelper, epcHelper, ueNodes, enbDevs, ueDevs, pathLossExponent);
    }
    else {
        // If setup isn't complete, keep checking after some time
        std::cout << "Setup not complete. Waiting for UE setup..." << std::endl;
        Simulator::Schedule(Seconds(0.5), &WaitForSetupAndTriggerHandover, lteHelper, epcHelper, ueNodes, enbDevs, ueDevs, pathLossExponent);
    }
}

void CalculateConnectedDevices(Ptr<LteEnbNetDevice> enbDevice, NetDeviceContainer ueDevs) {
    if (!enbDevice) {
        std::cerr << "[ERROR] Invalid eNB device!" << std::endl;
        return;
    }

    // Manually count connected UEs
    uint16_t connectedUes = 0;
    for (uint32_t i = 0; i < ueDevs.GetN(); ++i) {
        Ptr<LteUeNetDevice> ueDevice = DynamicCast<LteUeNetDevice>(ueDevs.Get(i));
        if (ueDevice && ueDevice->GetRrc()->GetCellId() == enbDevice->GetCellId()) {
            connectedUes++;
        }
    }

    // Print the number of connected devices
    std::cout << "Number of connected devices for eNB " << enbDevice->GetCellId() << ":" << std::endl;
    std::cout << "  Connected UEs: " << connectedUes << std::endl;
    std::cout << std::endl;

    // Schedule the next calculation
    Simulator::Schedule(Seconds(1.0), &CalculateConnectedDevices, enbDevice, ueDevs);
}

void PrintUeConnectionStatus(NetDeviceContainer ueDevs, NetDeviceContainer enbDevs, Ipv4InterfaceContainer ueIpIface) {
    std::cout << "\n=== UE Connection Status ===" << std::endl;
    
    for (uint32_t i = 0; i < ueDevs.GetN(); ++i) {
        Ptr<LteUeNetDevice> ueDevice = DynamicCast<LteUeNetDevice>(ueDevs.Get(i));
        // Get the cell ID from the UE's RRC
        uint16_t cellId = ueDevice->GetRrc()->GetCellId();
        
        // Get UE position
        Ptr<MobilityModel> ueMobility = ueDevice->GetNode()->GetObject<MobilityModel>();
        Vector uePos = ueMobility->GetPosition();
        
        // Find connected eNB
        Ptr<LteEnbNetDevice> connectedEnb = nullptr;
        Vector enbPos;
        for (uint32_t j = 0; j < enbDevs.GetN(); ++j) {
            Ptr<LteEnbNetDevice> enbDevice = DynamicCast<LteEnbNetDevice>(enbDevs.Get(j));
            if (enbDevice->GetCellId() == cellId) {
                connectedEnb = enbDevice;
                Ptr<MobilityModel> enbMobility = enbDevice->GetNode()->GetObject<MobilityModel>();
                enbPos = enbMobility->GetPosition();
                break;
            }
        }
        
        double distance = -1;
        if (connectedEnb) {
            distance = sqrt(pow(uePos.x - enbPos.x, 2) + pow(uePos.y - enbPos.y, 2));
        }
        
        std::cout << "UE " << i << ":" << std::endl;
        std::cout << "  IP Address: " << ueIpIface.GetAddress(i) << std::endl;
        std::cout << "  Position: (" << uePos.x << ", " << uePos.y << ")" << std::endl;
        std::cout << "  Connected to eNB: " << (connectedEnb ? "Yes" : "No") << std::endl;
        if (connectedEnb) {
            std::cout << "  Connected eNB ID: " << cellId << std::endl;
            std::cout << "  Distance to eNB: " << distance << " meters" << std::endl;
            // Print RSRP using CalculateRsrp function (adjusted for LTE)
            double rsrp = CalculateRsrp(ueDevice->GetNode(), connectedEnb->GetNode(), connectedEnb->GetPhy(0)->GetTxPower(), 3.5); // LTE path loss exponent
            std::cout << "  RSRP: " << rsrp << " dBm" << std::endl;
        }
    }

    // Periodic check
    Simulator::Schedule(Seconds(1.0), &PrintUeConnectionStatus, ueDevs, enbDevs, ueIpIface);
}

void CalculateThroughput(Ptr<FlowMonitor> flowMonitor, Ptr<Ipv4FlowClassifier> classifier, Ipv4InterfaceContainer ueIpIface) {
    static double lastTime = 0.0;
    static std::map<FlowId, FlowMonitor::FlowStats> lastStats;
    double currentTime = Simulator::Now().GetSeconds();
    double interval = currentTime - lastTime;
    
    std::map<FlowId, FlowMonitor::FlowStats> stats = flowMonitor->GetFlowStats();
    
    // Create maps to store total bytes for each UE (both sent and received)
    std::map<uint32_t, uint64_t> ueSentBytes;
    std::map<uint32_t, uint64_t> ueReceivedBytes;
    
    for (const auto& iter : stats) {
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(iter.first);
        
        // Calculate bytes in the interval
        uint64_t currentTotalBytes = iter.second.rxBytes;
        uint64_t bytesInInterval = 0;
        
        if (lastStats.find(iter.first) != lastStats.end()) {
            bytesInInterval = currentTotalBytes - lastStats[iter.first].rxBytes;
        }
        
        // Find which UE this flow belongs to (either as sender or receiver)
        for (uint32_t i = 0; i < ueIpIface.GetN(); ++i) {
            if (t.sourceAddress == ueIpIface.GetAddress(i)) {
                ueSentBytes[i] += bytesInInterval;
            }
            if (t.destinationAddress == ueIpIface.GetAddress(i)) {
                ueReceivedBytes[i] += bytesInInterval;
            }
        }
    }
    
    // Print throughput for each UE
    for (uint32_t i = 0; i < ueIpIface.GetN(); ++i) {
        double sentThroughput = (ueSentBytes[i] * 8.0) / (interval * 1e6); // Mbps
        double receivedThroughput = (ueReceivedBytes[i] * 8.0) / (interval * 1e6); // Mbps
        
        std::cout << "Time: " << currentTime << "s - UE " << i 
                  << " (" << ueIpIface.GetAddress(i) << ")"
                  << " Sent: " << sentThroughput << " Mbps"
                  << " Received: " << receivedThroughput << " Mbps" << std::endl;
    }
    
    // Store current stats for next interval
    lastStats = stats;
    lastTime = currentTime;
    
    // Schedule the next throughput calculation
    Simulator::Schedule(Seconds(1.0), &CalculateThroughput, flowMonitor, classifier, ueIpIface);
}

class LteGymEnv : public OpenGymEnv
{
public:
  LteGymEnv(Ptr<LteHelper> lteHelper, NetDeviceContainer enbDevs, 
            NetDeviceContainer ueDevs, double pathLossExponent,
            Ptr<FlowMonitor> flowMonitor, Ptr<Ipv4FlowClassifier> classifier, Ipv4InterfaceContainer ueIpIface);

  // OpenGym interface implementation
  Ptr<OpenGymSpace> GetObservationSpace();
  Ptr<OpenGymSpace> GetActionSpace();
  Ptr<OpenGymDataContainer> GetObservation();
  float GetReward();
  bool GetGameOver();
  std::string GetExtraInfo();
  bool ExecuteActions(Ptr<OpenGymDataContainer> action);

private:
  // Helper functions
  void CalculateState();
  void UpdateMetrics();
  double CalculateEnergyEfficiency();
  double CalculateQoSMetric();
  double CalculatePrbDeviation();
  double CalculatePrbUtilization(Ptr<LteEnbNetDevice> enbDevice);
  double CalculateEnbThroughput(Ptr<LteEnbNetDevice> enb); // Add this if needed
  
  // Environment components
  Ptr<LteHelper> m_lteHelper;
  NetDeviceContainer m_enbDevs;
  NetDeviceContainer m_ueDevs;
  double m_pathLossExponent;

  // Flow monitor and classifier
  Ptr<FlowMonitor> m_flowMonitor;
  Ptr<Ipv4FlowClassifier> m_classifier;
  Ipv4InterfaceContainer m_ueIpIface;

  // State variables
  std::vector<double> m_powers;
  std::vector<double> m_cioValues;
  std::vector<double> m_prbUtilization;
  std::vector<double> m_throughput;

  // Reward weights
  const double m_w1 = 0.5; // Energy efficiency weight
  const double m_w2 = 0.3; // QoS weight
  const double m_w3 = 0.2; // PRB deviation weight

  // Action bounds
  const double m_maxPowerAdjustment = 3.0;  // dB
  const double m_maxCioAdjustment = 2.0;    // dB

  // Episode parameters
  const uint32_t m_maxSteps = 50;
  uint32_t m_currentStep;
};

LteGymEnv::LteGymEnv(Ptr<LteHelper> lteHelper, NetDeviceContainer enbDevs, 
                     NetDeviceContainer ueDevs, double pathLossExponent,
                     Ptr<FlowMonitor> flowMonitor, Ptr<Ipv4FlowClassifier> classifier, Ipv4InterfaceContainer ueIpIface)
  : m_lteHelper(lteHelper),
    m_enbDevs(enbDevs),
    m_ueDevs(ueDevs),
    m_pathLossExponent(pathLossExponent),
    m_flowMonitor(flowMonitor),
    m_classifier(classifier),
    m_ueIpIface(ueIpIface),
    m_currentStep(0)
{
  uint32_t numEnb = enbDevs.GetN();
  m_powers.resize(numEnb);
  m_cioValues.resize(numEnb);
  m_prbUtilization.resize(numEnb);
  m_throughput.resize(numEnb);

  CalculateState();
}

Ptr<OpenGymSpace> LteGymEnv::GetObservationSpace()
{
  uint32_t numEnb = m_enbDevs.GetN();
  std::vector<uint32_t> shape = {numEnb, 4}; // 4 features per SBS
  std::string dtype = "float"; 
  std::cout << "Observation Space: [";
  for (size_t i = 0; i < shape.size(); ++i) {
      std::cout << shape[i];
      if (i < shape.size() - 1) {
          std::cout << ", ";
      }
  }
  std::cout << "]" << std::endl;
  
  
  Ptr<OpenGymBoxSpace> space = CreateObject<OpenGymBoxSpace>(0.0, 100.0, shape, dtype);
  return space;
}

Ptr<OpenGymSpace> LteGymEnv::GetActionSpace()
{
  uint32_t numEnb = m_enbDevs.GetN();
  std::vector<uint32_t> shape = {numEnb, 2}; // Power and CIO adjustments
  std::string dtype = TypeNameGet<float>(); 
  std::cout << "Action Space: [";
  for (size_t i = 0; i < shape.size(); ++i) {
      std::cout << shape[i];
      if (i < shape.size() - 1) {
          std::cout << ", ";
      }
  }
  std::cout << "]" << std::endl;
  
  // Action space bounds: [-max_adjustment, +max_adjustment]
  Ptr<OpenGymBoxSpace> space = CreateObject<OpenGymBoxSpace>(-m_maxPowerAdjustment, 
                                                            m_maxPowerAdjustment, 
                                                            shape, dtype);
  return space;
}

Ptr<OpenGymDataContainer> LteGymEnv::GetObservation()
{
  uint32_t numEnb = m_enbDevs.GetN();
  std::vector<uint32_t> shape = {numEnb, 4};

  std :: cout << "NumEnb: " << numEnb << std::endl;
  
  Ptr<OpenGymBoxContainer<double>> box = CreateObject<OpenGymBoxContainer<double>>(shape);
  
  // Pack state variables into observation
  for (uint32_t i = 0; i < numEnb; ++i) {
    box->AddValue(m_powers[i]);
    box->AddValue(m_cioValues[i]);
    box->AddValue(m_prbUtilization[i]);
    box->AddValue(m_throughput[i]);
  }

  std :: cout << "Observation: " << box << std::endl;
  
  return box;
}

float LteGymEnv::GetReward() {
    double r_ee = CalculateEnergyEfficiency()*100;
    // Convert RSRP to positive metric where higher is better
    double r_qos = (CalculateQoSMetric() + 100)/100; // Assuming RSRP in dBm
    // Convert PRB deviation to efficiency (1 - normalized_deviation)
    double r_prb = 1 - CalculatePrbDeviation()/100; // Assuming percentages
    
    // Linear combination with weights
    double reward = (m_w1 * r_ee + m_w2 * r_qos + m_w3 * r_prb) / (m_w1 + m_w2 + m_w3);
    
    std::cout << "Reward components: EE=" << r_ee << " QoS=" << r_qos 
              << " PRB=" << r_prb << " Total=" << reward << std::endl;
    
    return reward;
}

bool LteGymEnv::GetGameOver()
{
  std :: cout << "Current Step: " << m_currentStep << std::endl;
  return m_currentStep >= m_maxSteps;
}

std::string LteGymEnv::GetExtraInfo()
{
  std::string info = "Step: " + std::to_string(m_currentStep);
  return info;
}

bool LteGymEnv::ExecuteActions(Ptr<OpenGymDataContainer> action)
{
  Ptr<OpenGymBoxContainer<float>> box = DynamicCast<OpenGymBoxContainer<float>>(action);
  uint32_t numEnb = m_enbDevs.GetN();
  std :: cout << "Execute Actions: " << action << std::endl;
  
  // Apply power and CIO adjustments
  for (uint32_t i = 0; i < numEnb; ++i) {
    std :: cout << "i: " << i << std::endl;
    // Get power adjustment
    double powerAdj = box->GetValue(i * 2);
    Ptr<LteEnbNetDevice> enb = DynamicCast<LteEnbNetDevice>(m_enbDevs.Get(i));
    std :: cout << "Power Adjustment: " << powerAdj << std::endl;
    double currentPower = enb->GetPhy()->GetTxPower();
    double targetPower = std::max(minPowerDbm, std::min(maxPowerDbm, currentPower + powerAdj)); // Clamp to [10, 40]
    enb->GetPhy()->SetTxPower(targetPower);
    
    // Get CIO adjustment
    double cioAdj = box->GetValue(i * 2 + 1);
    std :: cout << "CIO Adjustment: " << cioAdj << std::endl;
    cioValues[i] += cioAdj;
    cioValues[i] = std::max(minCio, std::min(maxCio, cioValues[i])); // Clamp to [-10, 10]
  }

  std :: cout << "Actions: " << action << std::endl;
  
  // Update state after applying actions
  CalculateState();
  m_currentStep++;
  
  return true;
}

void ScheduleNextStateRead(double envStepTime, Ptr<OpenGymInterface> openGym)
{
  Simulator::Schedule (Seconds(envStepTime), &ScheduleNextStateRead, envStepTime, openGym);
  std :: cout << "Schedule Next State Read: " << envStepTime << std::endl;
  openGym->NotifyCurrentState();
}

// Add these helper function implementations
double LteGymEnv::CalculatePrbUtilization(Ptr<LteEnbNetDevice> enbDevice)
{
    if (!enbDevice) {
        std::cerr << "[ERROR] Invalid eNB device!" << std::endl;
        return 0;
    }

    // Manually count connected UEs
    float connectedUes = 0;
    for (uint32_t i = 0; i < m_ueDevs.GetN(); ++i) {
        Ptr<LteUeNetDevice> ueDevice = DynamicCast<LteUeNetDevice>(m_ueDevs.Get(i));
        if (ueDevice && ueDevice->GetRrc()->GetCellId() == enbDevice->GetCellId()) {
            connectedUes++;
        }
    }

    float prbUtilization = (connectedUes / m_ueDevs.GetN()) * 100;
    return prbUtilization;
}

double LteGymEnv::CalculateEnbThroughput(Ptr<LteEnbNetDevice> enb)
{
    static std::map<uint32_t, std::pair<double, uint64_t>> lastMeasurements;
    double currentTime = Simulator::Now().GetSeconds();
    uint16_t cellId = enb->GetCellId();
    
    // Get flow statistics
    std::map<FlowId, FlowMonitor::FlowStats> stats = m_flowMonitor->GetFlowStats();
    uint64_t currentTotalBytes = 0;
    
    // Get connected UE IPs
    std::vector<Ipv4Address> connectedUeIps;
    for (uint32_t i = 0; i < m_ueDevs.GetN(); ++i) {
        Ptr<LteUeNetDevice> ueDevice = DynamicCast<LteUeNetDevice>(m_ueDevs.Get(i));
        if (ueDevice && ueDevice->GetRrc()->GetCellId() == cellId) {
            connectedUeIps.push_back(m_ueIpIface.GetAddress(i));
        }
    }
    
    // Calculate current total bytes
    for (const auto& flow : stats) {
        Ipv4FlowClassifier::FiveTuple t = m_classifier->FindFlow(flow.first);
        
        for (const auto& ueIp : connectedUeIps) {
            if (t.sourceAddress == ueIp || t.destinationAddress == ueIp) {
                currentTotalBytes += (flow.second.rxBytes + flow.second.txBytes);
                break;
            }
        }
    }
    
    // Calculate throughput
    double throughput = 0.0;
    auto lastMeasurement = lastMeasurements.find(cellId);
    
    if (lastMeasurement != lastMeasurements.end()) {
        double timeDelta = currentTime - lastMeasurement->second.first;
        if (timeDelta > 0) {
            uint64_t bytesDelta = currentTotalBytes - lastMeasurement->second.second;
            throughput = (bytesDelta * 8.0) / (timeDelta * 1e6); // Convert to Mbps
        }
    }
    
    // Update last measurements
    lastMeasurements[cellId] = std::make_pair(currentTime, currentTotalBytes);
    
    // Add safety check for unreasonable values
    if (throughput > 1000) { // 1000 Mbps as a reasonable maximum
        std::cout << "Warning: Unusually high throughput detected for cell " << cellId 
                  << ": " << throughput << " Mbps" << std::endl;
        throughput = 0.0; // Reset to 0 or last known good value
    }
    
    return throughput;
}

// Helper functions implementation
void LteGymEnv::CalculateState()
{
  uint32_t numEnb = m_enbDevs.GetN();
  
  for (uint32_t i = 0; i < numEnb; ++i) {
    Ptr<LteEnbNetDevice> enb = DynamicCast<LteEnbNetDevice>(m_enbDevs.Get(i));
    
    // Update power
    m_powers[i] = enb->GetPhy()->GetTxPower();
    
    // Update CIO
    m_cioValues[i] = cioValues[i];
    
    // Calculate PRB utilization
    // Note: You'll need to implement PRB calculation based on your scheduler
    m_prbUtilization[i] = CalculatePrbUtilization(enb);
    
    // Calculate throughput
    m_throughput[i] = CalculateEnbThroughput(enb);
  }
}

double LteGymEnv::CalculateEnergyEfficiency()
{
  double totalThroughput = 0.0;
  double totalPower = 0.0;
  
  for (uint32_t i = 0; i < m_enbDevs.GetN(); ++i) {
    totalThroughput += m_throughput[i];
    totalPower += m_powers[i];
  }

  std :: cout << "Energy Efficiency: " << totalThroughput / totalPower << std::endl;
  
  return totalThroughput / totalPower;
}

double LteGymEnv::CalculateQoSMetric()
{
  double totalRsrp = 0.0;
  uint32_t numUes = m_ueDevs.GetN();
  
  for (uint32_t i = 0; i < numUes; ++i) {
    Ptr<LteUeNetDevice> ue = DynamicCast<LteUeNetDevice>(m_ueDevs.Get(i));
    uint16_t cellId = ue->GetRrc()->GetCellId();
    
    // Find connected eNB
    Ptr<LteEnbNetDevice> connectedEnb = nullptr;
    for (uint32_t j = 0; j < m_enbDevs.GetN(); ++j) {
      Ptr<LteEnbNetDevice> enb = DynamicCast<LteEnbNetDevice>(m_enbDevs.Get(j));
      if (enb->GetCellId() == cellId) {
        connectedEnb = enb;
        break;
      }
    }
    
    if (connectedEnb) {
      double rsrp = CalculateRsrp(ue->GetNode(), connectedEnb->GetNode(),
                                 connectedEnb->GetPhy()->GetTxPower(),
                                 m_pathLossExponent);
      totalRsrp += rsrp;
    }
  }

  std :: cout << "QoS Metric: " << totalRsrp / numUes << std::endl;
  
  return totalRsrp / numUes;
}

double LteGymEnv::CalculatePrbDeviation()
{
  double meanPrb = 0.0;
  uint32_t numEnb = m_enbDevs.GetN();
  
  // Calculate mean PRB
  for (uint32_t i = 0; i < numEnb; ++i) {
    meanPrb += m_prbUtilization[i];
  }
  meanPrb /= numEnb;
  
  // Calculate deviation
  double totalDeviation = 0.0;
  for (uint32_t i = 0; i < numEnb; ++i) {
    totalDeviation += std::abs(m_prbUtilization[i] - meanPrb);
  }
  std :: cout << "PRB Deviation: " << totalDeviation / numEnb << std::endl;
  return totalDeviation / numEnb;
}

int main(int argc, char *argv[]) {
    double simTime = 60;
    int numEnb = 10;  
    int numUes = 10;
    double minSpeed = 1.0;
    double maxSpeed = 3.0;
    double pathLossExponent = 3.5;
    double maxLength = 1000.0; //length of simulation environment

    std::cout << "Starting LTE simulation with " << numEnb 
              << " eNBs and " << numUes << " UEs." << std::endl;

    NodeContainer enbNodes;
    enbNodes.Create(numEnb);
    NodeContainer ueNodes;
    ueNodes.Create(numUes);

    RandomizeCioValues(numEnb, minCio, maxCio);

    // Create LTE helper
    Ptr<LteHelper> lteHelper = CreateObject<LteHelper>();
    Ptr<PointToPointEpcHelper> epcHelper = CreateObject<PointToPointEpcHelper>();
    lteHelper->SetEpcHelper(epcHelper);
    

    // Set default scheduler (To assign resSurces (PRBs) to UEs)
    lteHelper->SetSchedulerType("ns3::RrFfMacScheduler");
    // lteHelper->SetSchedulerType("ns3::TdTbfqFfMacScheduler"); // Time Domain Transmission

    // Configure the physical layer
    Config::SetDefault("ns3::LteEnbPhy::TxPower", DoubleValue(30.0));
    Config::SetDefault("ns3::LteUePhy::TxPower", DoubleValue(23.0));

    // Setup mobility for eNBs
    MobilityHelper enbMobility;
    enbMobility.SetPositionAllocator("ns3::RandomBoxPositionAllocator",
                                    "X", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=" + std::to_string(maxLength) + "]"),
                                    "Y", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=" + std::to_string(maxLength) + "]"),
                                    "Z", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
    enbMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    enbMobility.Install(enbNodes);

    // Setup mobility for UEs
    MobilityHelper ueMobility;
    ueMobility.SetPositionAllocator("ns3::RandomBoxPositionAllocator",
                                   "X", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=" + std::to_string(maxLength) + "]"),
                                   "Y", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=" + std::to_string(maxLength) + "]"),
                                   "Z", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
    ueMobility.SetMobilityModel("ns3::ConstantVelocityMobilityModel");
    ueMobility.Install(ueNodes);

    // Set random speed and direction for each UE
    for (uint32_t i = 0; i < ueNodes.GetN(); ++i) {
        Ptr<ConstantVelocityMobilityModel> mobility = 
            ueNodes.Get(i)->GetObject<ConstantVelocityMobilityModel>();

        double speed = minSpeed + (maxSpeed - minSpeed) * ((double) rand() / RAND_MAX);
        double direction = (2 * M_PI) * ((double) rand() / RAND_MAX);

        mobility->SetVelocity(Vector(speed * cos(direction), speed * sin(direction), 0));
        std::cout << "Set UE " << i << " speed to " << speed << " m/s, direction to " << direction << " radians." << std::endl;
    }

    checkUEPosition(ueNodes);

    // No automatic handover
    lteHelper->SetHandoverAlgorithmType("ns3::NoOpHandoverAlgorithm");
    // lteHelper->SetHandoverAlgorithmType("ns3::A3RsrpHandoverAlgorithm");
    // lteHelper->SetHandoverAlgorithmAttribute("Hysteresis", DoubleValue(0.5));
    // lteHelper->SetHandoverAlgorithmAttribute("TimeToTrigger", TimeValue(Seconds(0.1)));

    // Install LTE Devices
    NetDeviceContainer enbDevs = lteHelper->InstallEnbDevice(enbNodes);
    NetDeviceContainer ueDevs = lteHelper->InstallUeDevice(ueNodes);

    // Set random transmission power for each eNB
    for (uint32_t i = 0; i < enbDevs.GetN(); ++i) {
        Ptr<LteEnbNetDevice> enb = DynamicCast<LteEnbNetDevice>(enbDevs.Get(i));
        double power = minPowerDbm + (maxPowerDbm - minPowerDbm) * ((double) rand() / RAND_MAX);
        enb->GetPhy()->SetTxPower(power);
    }

    // Add X2 interface
    lteHelper->AddX2Interface(enbNodes);
    std::cout << "X2 interface added between eNBs" << std::endl;

    // Install the IP stack on the UEs
    InternetStackHelper internet;
    internet.Install(ueNodes);

    // Assign IP addresses to UEs
    Ipv4InterfaceContainer ueIpIface;
    ueIpIface = epcHelper->AssignUeIpv4Address(NetDeviceContainer(ueDevs));

    // Set default gateway for UEs
    Ipv4StaticRoutingHelper ipv4RoutingHelper;
    for (uint32_t u = 0; u < ueNodes.GetN(); ++u) {
        Ptr<Node> ueNode = ueNodes.Get(u);
        Ptr<Ipv4StaticRouting> ueStaticRouting = ipv4RoutingHelper.GetStaticRouting(ueNode->GetObject<Ipv4>());
        ueStaticRouting->SetDefaultRoute(epcHelper->GetUeDefaultGatewayAddress(), 1);
    }

    // Attach UEs to the closest eNB
    lteHelper->AttachToClosestEnb(ueDevs, enbDevs);
    std::cout << "UEs attached to the closest eNB." << std::endl;

    //Setup and configure bearers
    // Simulator::Schedule(Seconds(0.5), [&]() {
    //     EpsBearer bearer(EpsBearer::NGBR_VIDEO_TCP_DEFAULT);
    //     lteHelper->ActivateDataRadioBearer(ueDevs, bearer);
    // });
    
    PrintUeConnectionStatus(ueDevs, enbDevs, ueIpIface);
    PrintEnbMeasurements(enbDevs);
    PrintConnectedEnbRsrp(enbDevs, ueDevs, pathLossExponent, lteHelper);

    // Setup applications
    uint16_t serverPort = 8080;
    UdpServerHelper server(serverPort);
    // //install UDP server on all UES
    // for (uint32_t i = 0; i < ueNodes.GetN(); ++i) {
    //     ApplicationContainer serverApp = server.Install(ueNodes.Get(i));
    //     serverApp.Start(Seconds(0.0));
    //     serverApp.Stop(Seconds(simTime));
    //     std::cout << "Installing UDP server on UE " << i << " (IP: " << ueIpIface.GetAddress(i) 
    //               << ") port " << serverPort << std::endl;
    // }

    ApplicationContainer serverApp = server.Install(ueNodes.Get(0));
    
    std::cout << "Installing UDP server on UE " << 0 << " (IP: " << ueIpIface.GetAddress(0) 
              << ") port " << serverPort << std::endl;
    serverApp.Start(Seconds(0.0));
    serverApp.Stop(Seconds(simTime));

    // Install UDP clients
    ApplicationContainer clientApps;
    for (uint32_t i = 1; i < ueNodes.GetN(); ++i) {
        UdpClientHelper client(ueIpIface.GetAddress(0), serverPort);
        client.SetAttribute("MaxPackets", UintegerValue(0)); // Limit packets 0
        client.SetAttribute("Interval", TimeValue(MilliSeconds(100))); // Less frequent 100
        client.SetAttribute("PacketSize", UintegerValue(1024)); // Smaller packets 1024
        
        ApplicationContainer tempApp = client.Install(ueNodes.Get(i));
        clientApps.Add(tempApp);
        
        std :: cout << "Installing UDP client on UE " << i << " (IP: " << ueIpIface.GetAddress(i) 
                    << ") to send to UE " << i-1 << std::endl;
    }
    clientApps.Start(Seconds(1.0));
    clientApps.Stop(Seconds(simTime));

    // Setup flow monitor
    Ptr<FlowMonitor> flowMonitor;
    FlowMonitorHelper flowHelper;
    flowMonitor = flowHelper.InstallAll();
    flowHelper.SerializeToXmlFile("lte-simulation-flow.xml", true, true);

    // After the simulation, calculate and print the throughput
    flowMonitor->CheckForLostPackets();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(flowHelper.GetClassifier());
    // Start the throughput calculation function
    // Schedule the first throughput calculation
    Simulator::Schedule(Seconds(1.0), &CalculateThroughput, flowMonitor, classifier, ueIpIface);

    WaitForSetupAndTriggerHandover(lteHelper, epcHelper, ueNodes, enbDevs, ueDevs, pathLossExponent);

    //calculatePBU
    for (uint32_t i = 0; i < enbDevs.GetN(); ++i) {
        Ptr<LteEnbNetDevice> enbDevice = DynamicCast<LteEnbNetDevice>(enbDevs.Get(i));
        Simulator::Schedule(Seconds(1.0), &CalculateConnectedDevices, enbDevice, ueDevs);
    }

    // In your main() function, after installing mobility and before Simulator::Run()
    // Enable packet metadata to see data flow
    // Create the animation XML file

    // lteHelper->EnableDlPhyTraces();

    // // Enable packet tracing
    // lteHelper->EnablePhyTraces();
    // lteHelper->EnableMacTraces();
    // lteHelper->EnableRlcTraces();
    // lteHelper->EnablePdcpTraces();
    // AnimationInterface anim("lte-simulation.xml");

    // anim.EnablePacketMetadata(true);
    
    // // Set node colors and descriptions
    // for (uint32_t i = 0; i < ueNodes.GetN(); ++i) {
    //     anim.UpdateNodeDescription(ueNodes.Get(i), "UE-" + std::to_string(i));
    //     anim.UpdateNodeColor(ueNodes.Get(i), 0, 0, 255); // Blue for UEs
    //     anim.UpdateNodeSize(i, 10, 10); // Make UEs visible
    // }

    // for (uint32_t i = 0; i < enbNodes.GetN(); ++i) {
    //     anim.UpdateNodeDescription(enbNodes.Get(i), "SBS-" + std::to_string(i+1));
    //     anim.UpdateNodeColor(enbNodes.Get(i), 255, 0, 0); // Red for SBS
    //     anim.UpdateNodeSize(i + ueNodes.GetN(), 20, 20); // Make SBS bigger than UEs
    // }

    // Create OpenGym Env
    Ptr<OpenGymInterface> openGymInterface = CreateObject<OpenGymInterface>(5555);
    Ptr<LteGymEnv> lteEnv = CreateObject<LteGymEnv>(lteHelper, enbDevs, ueDevs, pathLossExponent, flowMonitor, classifier, ueIpIface);
    
    openGymInterface->SetGetActionSpaceCb(MakeCallback(&LteGymEnv::GetActionSpace, lteEnv));
    openGymInterface->SetGetObservationSpaceCb(MakeCallback(&LteGymEnv::GetObservationSpace, lteEnv));
    openGymInterface->SetGetGameOverCb(MakeCallback(&LteGymEnv::GetGameOver, lteEnv));
    openGymInterface->SetGetObservationCb(MakeCallback(&LteGymEnv::GetObservation, lteEnv));
    openGymInterface->SetGetRewardCb(MakeCallback(&LteGymEnv::GetReward, lteEnv));
    openGymInterface->SetGetExtraInfoCb(MakeCallback(&LteGymEnv::GetExtraInfo, lteEnv));
    openGymInterface->SetExecuteActionsCb(MakeCallback(&LteGymEnv::ExecuteActions, lteEnv));
    Simulator::Schedule (Seconds(0.0), &ScheduleNextStateRead, 1.0, openGymInterface);

    // // Set update rate for smoother animation
    // anim.SetMobilityPollInterval(Seconds(0.01));
    
    // // Enable packet metadata with custom settings
    // anim.EnablePacketMetadata(true);
    // anim.SetMaxPktsPerTraceFile(1000000);


    Simulator::Stop(Seconds(simTime));
    Simulator::Run();
    openGymInterface->NotifySimulationEnd();
    Simulator::Destroy();

    std::cout << "Simulation finished." << std::endl;
    return 0;
}