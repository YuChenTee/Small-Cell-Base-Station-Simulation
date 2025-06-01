#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/lte-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/point-to-point-helper.h"
#include "ns3/opengym-module.h"
#include <ctime> 

using namespace ns3;

std::vector<double> cioValues; // Store CIO values for each SBS

double CalculateRsrp(Ptr<Node> ueNode, Ptr<Node> enbNode, double txPowerDbm, double pathLossExponent) {
    Ptr<MobilityModel> ueMobility = ueNode->GetObject<MobilityModel>();
    Ptr<MobilityModel> enbMobility = enbNode->GetObject<MobilityModel>();
    double distance = ueMobility->GetDistanceFrom(enbMobility);
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

void checkUEPosition(NodeContainer& ueNodes) {
    for (uint32_t i = 0; i < ueNodes.GetN(); ++i) {
        Ptr<ConstantVelocityMobilityModel> mobility = 
            ueNodes.Get(i)->GetObject<ConstantVelocityMobilityModel>();
        std::cout << "UE " << i << " position: " << mobility->GetPosition() 
                << ", velocity: " << mobility->GetVelocity() << std::endl;
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

        if (targetEnbDevice && targetCellId != currentCellId && maxAdjustedRsrp > (servingRsrp + servingCio)) {
            // Verify UE is still in valid state before triggering handover
            if (ueRrc->GetState() == LteUeRrc::CONNECTED_NORMALLY && 
                ueRrc->GetRnti() != 0) {
                
                std::cout << "\n[INFO] Triggering handover for UE " << i 
                          << "\n  From CellId: " << currentCellId
                          << " (RSRP + CIO: " << servingRsrp + servingCio << " dB)"
                          << "\n  To CellId: " << targetCellId 
                          << " (RSRP + CIO: " << maxAdjustedRsrp << " dB)"
                          << "\n  Improvement: " << maxAdjustedRsrp - (servingRsrp + servingCio) << " dB" << std::endl;

                lteHelper->HandoverRequest(Seconds(0), ueDevice, servingEnb, targetCellId);
                std::cout << "[DEBUG] Handover request sent successfully" << std::endl;
            } else {
                std::cout << "[WARNING] Skipping handover for UE " << i << " due to invalid state" << std::endl;
            }
        }
    }

    // Schedule the next handover decision
    const double HANDOVER_CHECK_INTERVAL = 0.2; // Check every 1 second
    Simulator::Schedule(Seconds(HANDOVER_CHECK_INTERVAL), &HandoverDecision, 
                        lteHelper, epcHelper, ueNodes, enbDevs, ueDevs, pathLossExponent);

    std::cout << "\n[INFO] Handover decision completed. Next check scheduled in " 
              << HANDOVER_CHECK_INTERVAL << " seconds.\n" << std::endl;
}

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

class LteGymEnv : public OpenGymEnv
{
public:
    LteGymEnv(Ptr<LteHelper> lteHelper, NetDeviceContainer enbDevs, 
              NetDeviceContainer ueDevs, double pathLossExponent);

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
    double CalculateAveragePowerConsumption();
    double CalculateQoSMetric();
    double CalculatePrbDeviation();
    double CalculatePrbUtilization(Ptr<LteEnbNetDevice> enbDevice);
    double CalculateEnbThroughput(Ptr<LteEnbNetDevice> enb); 
    double ScaleEnergyEfficiency(double ee);
    double ScaleAveragePowerConsumption(double avgPower);
    double ScaleQoSMetric(double avgRsrp);
    double ScalePrbDeviation(double prbDeviation);
    
    // Environment components
    Ptr<LteHelper> m_lteHelper;
    NetDeviceContainer m_enbDevs;
    NetDeviceContainer m_ueDevs;
    double m_pathLossExponent;

    // State variables
    std::vector<double> m_powers;
    std::vector<double> m_cioValues;
    std::vector<double> m_prbUtilization;

    // Reward weights
    const double m_w_power = 0.3; // Power consumption weight
    const double m_w_qos = 0.4;   // QoS weight
    const double m_w_prb = 0.3;   // PRB deviation weight

    // Average power consumption thresholds
    const double m_power_excellent = 25;  // Excellent power consumption (25)
    const double m_power_poor = 40;    // Poor power consumption (35)

    // QoS thresholds (RSRP in dBm)
    const double m_rsrp_excellent = -45.0;  // Excellent signal strength (-40)
    const double m_rsrp_poor = -75.0;      // Poor signal strength (-70)

    // PRB deviation targets
    const double m_prb_excellent = 8;   // Excellent PRB Deviation (10)
    const double m_prb_poor = 20;   // Poor PRB Deviation (20)

    // Action bounds
    const double m_maxPowerAdjustment = 3.0;  // dB
    const double m_maxCioAdjustment = 2.0;    // dB

    // Episode parameters
    const uint32_t m_maxSteps = 100;
    uint32_t m_currentStep;
};

LteGymEnv::LteGymEnv(Ptr<LteHelper> lteHelper, NetDeviceContainer enbDevs, 
                     NetDeviceContainer ueDevs, double pathLossExponent)
  : m_lteHelper(lteHelper),
    m_enbDevs(enbDevs),
    m_ueDevs(ueDevs),
    m_pathLossExponent(pathLossExponent),
    m_currentStep(0)
{
    uint32_t numEnb = enbDevs.GetN();
    m_powers.resize(numEnb);
    m_cioValues.resize(numEnb);
    m_prbUtilization.resize(numEnb);

    CalculateState();
}

Ptr<OpenGymSpace> LteGymEnv::GetObservationSpace()
{
    uint32_t numEnb = m_enbDevs.GetN();
    std::vector<uint32_t> shape = {numEnb, 3}; // 3 features per SBS
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
    std::vector<uint32_t> shape = {numEnb, 3};
    std::cout << "NumEnb: " << numEnb << std::endl;

    Ptr<OpenGymBoxContainer<double>> box = CreateObject<OpenGymBoxContainer<double>>(shape);
    
    // Pack state variables into observation
    for (uint32_t i = 0; i < numEnb; ++i) {
        box->AddValue(m_powers[i]);
        box->AddValue(m_cioValues[i]);
        box->AddValue(m_prbUtilization[i]);
    }
    std::cout << "Observation: " << box << std::endl;  
    return box;
}

float LteGymEnv::GetReward() {
    double avg_power = CalculateAveragePowerConsumption();
    double r_power = ScaleAveragePowerConsumption(avg_power);
    
    // 2. QoS Component
    double avg_rsrp = CalculateQoSMetric();
    double r_qos = ScaleQoSMetric(avg_rsrp);
    
    // 3. PRB Utilization Component
    double prb_deviation = CalculatePrbDeviation();
    double r_prb = ScalePrbDeviation(prb_deviation);
    
    // Combine components with weights
    double total_reward = (m_w_power * r_power + 
                            m_w_qos * r_qos + 
                            m_w_prb * r_prb);
    
    // Log components for debugging
    std::cout << "\nReward Components:"
                << "\n  Avg Power: " << avg_power << " dBm"
                << "\n  Power Scaled: " << r_power
                << "\n  Avg RSRP: " << avg_rsrp << " dBm"
                << "\n  QoS Scaled: " << r_qos
                << "\n  PRB Deviation: " << prb_deviation << "%"
                << "\n  PRB Scaled: " << r_prb
                << "\n  Total Reward: " << total_reward
                << std::endl;
    
    return total_reward;
}

bool LteGymEnv::GetGameOver()
{
    std::cout << "Current Step: " << m_currentStep << std::endl;
    return m_currentStep >= m_maxSteps;
}

std::string LteGymEnv::GetExtraInfo()
{
    double avg_power = CalculateAveragePowerConsumption();
    double r_power = ScaleAveragePowerConsumption(avg_power);
    
    // 2. QoS Component
    double avg_rsrp = CalculateQoSMetric();
    double r_qos = ScaleQoSMetric(avg_rsrp);
    
    // 3. PRB Utilization Component
    double prb_deviation = CalculatePrbDeviation();
    double r_prb = ScalePrbDeviation(prb_deviation);

    std::ostringstream oss;
    oss << "Power:" << avg_power
        << ",QoS:" << avg_rsrp
        << ",PRB:" << prb_deviation
        << ",Power(Scaled):" << r_power
        << ",QoS(Scaled):" << r_qos
        << ",PRB(Scaled):" << r_prb
        << ",Power_excellent_threshold:" << 25
        << ",Power_poor_threshold:" << 35
        << ",QoS_excellent_threshold:" << -60
        << ",QoS_poor_threshold:" << -100
        << ",PRB_excellent_threshold:" << 15
        << ",PRB_poor_threshold:" << 30
        ;

    return oss.str();
}

bool LteGymEnv::ExecuteActions(Ptr<OpenGymDataContainer> action)
{
    Ptr<OpenGymBoxContainer<float>> box = DynamicCast<OpenGymBoxContainer<float>>(action);
    uint32_t numEnb = m_enbDevs.GetN();
    std::cout << "Execute Actions: " << action << std::endl;
    
    // Apply power and CIO adjustments
    for (uint32_t i = 0; i < numEnb; ++i) {
        std::cout << "i: " << i << std::endl;
        // Get power adjustment
        double powerAdj = box->GetValue(i * 2);
        Ptr<LteEnbNetDevice> enb = DynamicCast<LteEnbNetDevice>(m_enbDevs.Get(i));
        std::cout << "Power Adjustment: " << powerAdj << std::endl;
        enb->GetPhy()->SetTxPower(powerAdj);
        
        // Get CIO adjustment
        double cioAdj = box->GetValue(i * 2 + 1);
        std::cout << "CIO Adjustment: " << cioAdj << std::endl;
        cioValues[i] = cioAdj;
    }
    
    std::cout << "Actions: " << action << std::endl;

    // Update state after applying actions
    CalculateState();
    m_currentStep++;
    
    return true;
}

double LteGymEnv::ScaleAveragePowerConsumption(double power) {
    // Linear scaling for power consumption
    if (power <= m_power_excellent) return 1.0;
    if (power >= m_power_poor) return -1.0;
    
    return 2.0 * (power - m_power_poor) / 
        (m_power_excellent-m_power_poor) - 1.0;
}

double LteGymEnv::ScaleQoSMetric(double rsrp) {
    // Linear scaling between poor and excellent thresholds
    if (rsrp >= m_rsrp_excellent) return 1.0;
    if (rsrp <= m_rsrp_poor) return -1.0;
    
    return 2.0 * (rsrp - m_rsrp_poor) / 
            (m_rsrp_excellent - m_rsrp_poor) - 1.0;
}

double LteGymEnv::ScalePrbDeviation(double deviation) {
    // Linear scaling for PRB deviation between poor and excellent thresholds
    if (deviation <= m_prb_excellent) return 1.0;
    if (deviation >= m_prb_poor) return -1.0;
    
    return 2.0 * (deviation - m_prb_poor) / 
        (m_prb_excellent-m_prb_poor) - 1.0;
}

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
        m_prbUtilization[i] = CalculatePrbUtilization(enb);
    }
}

double LteGymEnv::CalculateAveragePowerConsumption()
{
    double totalPower = 0.0;
    
    for (uint32_t i = 0; i < m_enbDevs.GetN(); ++i) {
        totalPower += m_powers[i];
    }

    std::cout << "Average Power Consumption: " << totalPower / m_enbDevs.GetN() << std::endl;

    return totalPower / m_enbDevs.GetN();
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

    std::cout << "QoS Metric: " << totalRsrp / numUes << std::endl;
    
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
    std::cout << "PRB Deviation: " << totalDeviation / numEnb << std::endl;
    return totalDeviation / numEnb;
}

void ScheduleNextStateRead(double envStepTime, Ptr<OpenGymInterface> openGym)
{
    Simulator::Schedule(Seconds(envStepTime), &ScheduleNextStateRead, envStepTime, openGym);
    std::cout << "Schedule Next State Read: " << envStepTime << std::endl;
    openGym->NotifyCurrentState();
}

int main(int argc, char *argv[]) {
    double simTime = 60;
    int numEnb = 3;  
    int numUes = 30;
    double minSpeed = 1.0;
    double maxSpeed = 3.0;
    double pathLossExponent = 3.5;
    double intersiteDistance = 250;
    double width = intersiteDistance*4;
    double height = intersiteDistance*2;
    double powerOptions[] = {20.0, 30.0, 40.0}; // Power options for eNBs
    double cioOptions[] = {-10.0, 0.0, 10.0}; // CIO options for eNBs

    std::cout << "Starting LTE simulation with " << numEnb 
              << " eNBs and " << numUes << " UEs." << std::endl;

    // Default arguments defined by ns3gym, can set from python and use here
    CommandLine cmd;
    uint16_t openGymPort;    
    uint32_t seed;  
    cmd.AddValue("openGymPort", "Port for OpenGym-ZMQ", openGymPort);
    cmd.AddValue("simSeed", "Random seed value", seed);
    cmd.Parse(argc, argv);
    ns3::SeedManager::SetSeed(seed); // so that random variables are different everytime

    NodeContainer enbNodes;
    enbNodes.Create(numEnb);
    NodeContainer ueNodes;
    ueNodes.Create(numUes);

    // Create LTE helper
    Ptr<LteHelper> lteHelper = CreateObject<LteHelper>();
    Ptr<PointToPointEpcHelper> epcHelper = CreateObject<PointToPointEpcHelper>();
    lteHelper->SetEpcHelper(epcHelper);
    
    // Set default scheduler (To assign resSurces (PRBs) to UEs)
    lteHelper->SetSchedulerType("ns3::RrFfMacScheduler");

    // Configure the physical layer
    Config::SetDefault("ns3::LteEnbPhy::TxPower", DoubleValue(30.0));
    Config::SetDefault("ns3::LteUePhy::TxPower", DoubleValue(23.0));

    // Setup fixed positions for eNBs
    Ptr<ListPositionAllocator> enbPositionAlloc = CreateObject<ListPositionAllocator>();
    enbPositionAlloc->Add(Vector(intersiteDistance, height/2, 0.0));
    enbPositionAlloc->Add(Vector(intersiteDistance*2, height/2, 0.0));
    enbPositionAlloc->Add(Vector(intersiteDistance*3, height/2, 0.0));

    MobilityHelper enbMobility;
    enbMobility.SetPositionAllocator(enbPositionAlloc);
    enbMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    enbMobility.Install(enbNodes);

    // Setup mobility for UEs
    MobilityHelper ueMobility;
    ueMobility.SetPositionAllocator("ns3::RandomBoxPositionAllocator",
                                   "X", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=" + std::to_string(width) + "]"),
                                   "Y", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=" + std::to_string(height) + "]"),
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

    // Install LTE Devices
    NetDeviceContainer enbDevs = lteHelper->InstallEnbDevice(enbNodes);
    NetDeviceContainer ueDevs = lteHelper->InstallUeDevice(ueNodes);

    // Set random transmission power for each eNB
    for (uint32_t i = 0; i < enbDevs.GetN(); ++i) {
        Ptr<LteEnbNetDevice> enb = DynamicCast<LteEnbNetDevice>(enbDevs.Get(i));
        // Pick a random index from the power options array
        int powerIdx = rand() % 3;  // Random index between 0, 1, or 2
        double power = powerOptions[powerIdx];
        enb->GetPhy()->SetTxPower(power);
    }

    // Set random CIO values for each eNB
    for (int i = 0; i < numEnb; ++i) {
        // Pick a random index from the cioOptions vector
        int cioIdx = rand() % 3;
        double cio = cioOptions[cioIdx];
        cioValues.push_back(cio);
        std::cout << "eNB " << i << " CIO value: " << cio << " dB" << std::endl;
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
    
    PrintUeConnectionStatus(ueDevs, enbDevs, ueIpIface);
    PrintEnbMeasurements(enbDevs);
    PrintConnectedEnbRsrp(enbDevs, ueDevs, pathLossExponent, lteHelper);

    // Setup applications
    uint16_t serverPort = 8080;
    UdpServerHelper server(serverPort);
    //install UDP server on all UES
    for (uint32_t i = 0; i < ueNodes.GetN(); ++i) {
        ApplicationContainer serverApp = server.Install(ueNodes.Get(i));
        serverApp.Start(Seconds(0.0));
        serverApp.Stop(Seconds(simTime));
        std::cout << "Installing UDP server on UE " << i << " (IP: " << ueIpIface.GetAddress(i) 
                  << ") port " << serverPort << std::endl;
    }

    // Install UDP clients
    ApplicationContainer clientApps;
    for (uint32_t i = 1; i < ueNodes.GetN(); ++i) {
        UdpClientHelper client(ueIpIface.GetAddress(i-1), serverPort);
        client.SetAttribute("MaxPackets", UintegerValue(0));             // Unlimited packets
        client.SetAttribute("Interval", TimeValue(MicroSeconds(100)));   // Higher frequency
        client.SetAttribute("PacketSize", UintegerValue(8192));         // Larger packets
        
        ApplicationContainer tempApp = client.Install(ueNodes.Get(i));
        clientApps.Add(tempApp);
        
        std::cout << "Installing UDP client on UE " << i << " (IP: " << ueIpIface.GetAddress(i) 
                << ") to send to UE " << i-1 << std::endl;
    }
    clientApps.Start(Seconds(0.0));
    clientApps.Stop(Seconds(simTime));

    Simulator::Schedule(Seconds(0.5), &WaitForSetupAndTriggerHandover, lteHelper, epcHelper, ueNodes, enbDevs, ueDevs, pathLossExponent);

    //calculatePBU
    for (uint32_t i = 0; i < enbDevs.GetN(); ++i) {
        Ptr<LteEnbNetDevice> enbDevice = DynamicCast<LteEnbNetDevice>(enbDevs.Get(i));
        Simulator::Schedule(Seconds(1.0), &CalculateConnectedDevices, enbDevice, ueDevs);
    }

    // Create OpenGym Env
    Ptr<OpenGymInterface> openGymInterface = CreateObject<OpenGymInterface>(5555);
    Ptr<LteGymEnv> lteEnv = CreateObject<LteGymEnv>(lteHelper, enbDevs, ueDevs, pathLossExponent);
    
    openGymInterface->SetGetActionSpaceCb(MakeCallback(&LteGymEnv::GetActionSpace, lteEnv));
    openGymInterface->SetGetObservationSpaceCb(MakeCallback(&LteGymEnv::GetObservationSpace, lteEnv));
    openGymInterface->SetGetGameOverCb(MakeCallback(&LteGymEnv::GetGameOver, lteEnv));
    openGymInterface->SetGetObservationCb(MakeCallback(&LteGymEnv::GetObservation, lteEnv));
    openGymInterface->SetGetRewardCb(MakeCallback(&LteGymEnv::GetReward, lteEnv));
    openGymInterface->SetGetExtraInfoCb(MakeCallback(&LteGymEnv::GetExtraInfo, lteEnv));
    openGymInterface->SetExecuteActionsCb(MakeCallback(&LteGymEnv::ExecuteActions, lteEnv));
    Simulator::Schedule (Seconds(0.0), &ScheduleNextStateRead, 0.2, openGymInterface);

    Simulator::Stop(Seconds(simTime));
    Simulator::Run();
    openGymInterface->NotifySimulationEnd();
    Simulator::Destroy();

    std::cout << "Simulation finished." << std::endl;
    return 0;
}