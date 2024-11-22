#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/nr-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/point-to-point-helper.h"
#include "ns3/eps-bearer-tag.h"
#include "ns3/netanim-module.h"
#include "ns3/flow-monitor-helper.h"
#include "ns3/ipv4-flow-classifier.h"
#include <ns3/antenna-module.h>

using namespace ns3;

std::vector<double> cioValues;
double minCio = -10.0;
double maxCio = 10.0;

double CalculateRsrp(Ptr<Node> ueNode, Ptr<Node> gnbNode, double txPowerDbm, double pathLossExponent) {
    Ptr<MobilityModel> ueMobility = ueNode->GetObject<MobilityModel>();
    Ptr<MobilityModel> gnbMobility = gnbNode->GetObject<MobilityModel>();
    double distance = ueMobility->GetDistanceFrom(gnbMobility);
    double txPowerWatts = pow(10.0, txPowerDbm / 10.0) / 1000.0;
    double rsrp = txPowerWatts / pow(distance, pathLossExponent);
    return 10 * log10(rsrp * 1000.0);
}

void PrintGnbMeasurements(NetDeviceContainer& gnbDevs, NodeContainer& ueNodes, double pathLossExponent) {
    std::cout << "Getting Transmission power for all gNBs..." << std::endl;
    for (uint32_t i = 0; i < gnbDevs.GetN(); ++i) {
        Ptr<NrGnbNetDevice> gnbDevice = DynamicCast<NrGnbNetDevice>(gnbDevs.Get(i));
        Ptr<NrGnbPhy> gnbPhy = gnbDevice->GetPhy(0);
        double txPowerDbm = gnbPhy->GetTxPower();
        std::cout << "gNB " << i << " transmission power: " << txPowerDbm << " dBm" << std::endl;
    }
}

void PrintConnectedGnbRsrp(NetDeviceContainer& gnbDevs, NetDeviceContainer& ueDevs, 
                          double pathLossExponent, Ptr<NrHelper> nrHelper) {
    
    std::cout << "Calculating RSRP for connected gNBs..." << std::endl;
    for (uint32_t i = 0; i < ueDevs.GetN(); ++i) {
        Ptr<NrUeNetDevice> ueDevice = DynamicCast<NrUeNetDevice>(ueDevs.Get(i));
        uint16_t cellId = ueDevice->GetRrc()->GetCellId();
        
        Ptr<NrGnbNetDevice> connectedGnb = nullptr;
        for (uint32_t j = 0; j < gnbDevs.GetN(); ++j) {
            Ptr<NrGnbNetDevice> gnbDevice = DynamicCast<NrGnbNetDevice>(gnbDevs.Get(j));
            if (gnbDevice->GetCellId() == cellId) {
                connectedGnb = gnbDevice;
                break;
            }
        }
        
        if (connectedGnb != nullptr) {
            Ptr<NrGnbPhy> gnbPhy = connectedGnb->GetPhy(0);
            double txPowerDbm = gnbPhy->GetTxPower();
            double rsrp = CalculateRsrp(ueDevs.Get(i)->GetNode(), 
                                      connectedGnb->GetNode(), 
                                      txPowerDbm, 
                                      pathLossExponent);
            std::cout << "RSRP for UE " << i << " to connected gNB (Cell ID " 
                     << cellId << "): " << rsrp << " dBm" << std::endl;
        }
    }
}

void RandomizeCioValues(int numGnb, double minCio, double maxCio) {
    cioValues.clear();
    for (int i = 0; i < numGnb; ++i) {
        double cio = minCio + (maxCio - minCio) * ((double) rand() / RAND_MAX);
        cioValues.push_back(cio);
        std::cout << "gNB " << i << " CIO value: " << cio << " dB" << std::endl;
    }
}

void ManualHandover(Ptr<NrHelper> nrHelper, Ptr<NrPointToPointEpcHelper> epcHelper, NetDeviceContainer& ueDevs, NetDeviceContainer& gnbDevs, uint32_t ueIndex, uint32_t targetGnbIndex, Ptr<NrGnbNetDevice> currentGnbDevice) {
    NS_ASSERT(nrHelper != nullptr);
    NS_ASSERT(ueDevs.GetN() > 0 && gnbDevs.GetN() > 0);

    // Ensure UE and target gNB are valid
    Ptr<NrUeNetDevice> ueDevice = DynamicCast<NrUeNetDevice>(ueDevs.Get(ueIndex));
    Ptr<NrGnbNetDevice> targetGnbDevice = DynamicCast<NrGnbNetDevice>(gnbDevs.Get(targetGnbIndex-1));

    if (!ueDevice || !targetGnbDevice) {
        std::cout << "[ERROR] Invalid UE or target gNB at indices " << ueIndex << " and " << targetGnbIndex << std::endl;
        return;
    }

    // Step 1: Attach the UE to the target gNB before releasing the previous connection
    nrHelper->AttachToGnb(ueDevice, targetGnbDevice);
    std::cout << "[INFO] UE attached to target gNB (CellId: " << targetGnbDevice->GetCellId() << ")." << std::endl;

    // Step 5: Trigger NAS signaling or bearer setup
    uint8_t qos = 5;  // Example QoS value

    // Create the bearer with the desired QoS
    NrEpsBearer bearer{NrEpsBearer::Qci(qos)};

    // Optionally, create a Traffic Flow Template (TFT) if needed
    Ptr<NrEpcTft> tft = new NrEpcTft();

    // Activate the dedicated EPS bearer
    uint8_t bearerId = nrHelper->ActivateDedicatedEpsBearer(ueDevice, bearer, tft);

    epcHelper->ActivateEpsBearer(ueDevice, bearerId, tft, bearer);

    std::cout << "[INFO] Bearer activated for the UE." << std::endl;

    // Step 2: Release any resources on the current serving gNB only after attaching to the target gNB
    Ptr<NrGnbMac> currentGnbMac = currentGnbDevice->GetMac(0);
    if (!currentGnbMac) {
        std::cout << "[ERROR] currentGnbMac is null, unable to release resources." << std::endl;
        return;
    }

    auto cmacSapProvider = currentGnbMac->GetGnbCmacSapProvider();
    if (!cmacSapProvider) {
        std::cout << "[ERROR] cmacSapProvider is null, unable to release logical channel." << std::endl;
        return;
    }

    Ptr<NrUeRrc> ueRrc = ueDevice->GetRrc();
    if (!ueRrc) {
        std::cout << "[ERROR] UE RRC is null, unable to release resources on the current gNB." << std::endl;
        return;
    }

    uint16_t rnti = ueRrc->GetRnti();
    if (rnti != 0) {
        cmacSapProvider->ReleaseLc(rnti, 1);  // Release logical channel on current gNB
        std::cout << "[INFO] Released resources on the current gNB (CellId: " << currentGnbDevice->GetCellId() << ")." << std::endl;
    } else {
        std::cout << "[ERROR] Invalid RNTI, unable to release resources on current gNB." << std::endl;
    }

    // Step 3: Disconnect UE RRC from the current serving gNB
    auto asSapProvider = ueRrc->GetAsSapProvider();
    if (asSapProvider) {
        asSapProvider->Disconnect();
        std::cout << "[INFO] UE RRC disconnected from current gNB (CellId: " << currentGnbDevice->GetCellId() << ")." << std::endl;
    } else {
        std::cout << "[ERROR] asSapProvider is null, unable to disconnect from current gNB." << std::endl;
    }
}
void
NrUeManager::PrepareHandover(uint16_t cellId)
{
    std::cout << "Function: PrepareHandover, CellId: " << cellId << std::endl;

    switch (m_state)
    {
    case CONNECTED_NORMALLY: {
        m_targetCellId = cellId;

        auto sourceComponentCarrier = DynamicCast<BandwidthPartGnb>(
            m_rrc->m_componentCarrierPhyConf.at(m_componentCarrierId));
        NS_ASSERT(m_rrc != nullptr && "RRC is null");
        NS_ASSERT(sourceComponentCarrier != nullptr && "Source component carrier is null");

        if (m_targetCellId == sourceComponentCarrier->GetCellId())
        {
            std::cerr << "Error: Target Cell ID matches Source Cell ID. Exiting." << std::endl;
            return;
        }

        if (m_rrc->HasCellId(cellId))
        {
            // Intra-gNB handover
            std::cout << "Intra-gNB handover for CellId: " << cellId << std::endl;

            uint8_t componentCarrierId = m_rrc->CellToComponentCarrierId(cellId);
            uint16_t rnti = m_rrc->AddUe(NrUeManager::HANDOVER_JOINING, componentCarrierId);
            NrGnbCmacSapProvider::AllocateNcRaPreambleReturnValue anrcrv =
                m_rrc->m_cmacSapProvider.at(componentCarrierId)->AllocateNcRaPreamble(rnti);

            if (!anrcrv.valid)
            {
                std::cerr << "Failed to allocate a preamble for non-contention-based RA. "
                          << "Cannot perform HO." << std::endl;
                return;
            }

            Ptr<NrUeManager> ueManager = m_rrc->GetUeManager(rnti);
            ueManager->SetSource(sourceComponentCarrier->GetCellId(), m_rnti);
            ueManager->SetImsi(m_imsi);

            // Setup data radio bearers
            for (auto& it : m_drbMap)
            {
                ueManager->SetupDataRadioBearer(it.second->m_epsBearer,
                                                it.second->m_epsBearerIdentity,
                                                it.second->m_gtpTeid,
                                                it.second->m_transportLayerAddress);
            }

            NrRrcSap::RrcConnectionReconfiguration handoverCommand =
                GetRrcConnectionReconfigurationForHandover(componentCarrierId);

            handoverCommand.mobilityControlInfo.newUeIdentity = rnti;
            handoverCommand.mobilityControlInfo.haveRachConfigDedicated = true;
            handoverCommand.mobilityControlInfo.rachConfigDedicated.raPreambleIndex =
                anrcrv.raPreambleId;
            handoverCommand.mobilityControlInfo.rachConfigDedicated.raPrachMaskIndex =
                anrcrv.raPrachMaskIndex;

            NrGnbCmacSapProvider::RachConfig rc =
                m_rrc->m_cmacSapProvider.at(componentCarrierId)->GetRachConfig();
            handoverCommand.mobilityControlInfo.radioResourceConfigCommon.rachConfigCommon
                .preambleInfo.numberOfRaPreambles = rc.numberOfRaPreambles;
            handoverCommand.mobilityControlInfo.radioResourceConfigCommon.rachConfigCommon
                .raSupervisionInfo.preambleTransMax = rc.preambleTransMax;
            handoverCommand.mobilityControlInfo.radioResourceConfigCommon.rachConfigCommon
                .raSupervisionInfo.raResponseWindowSize = rc.raResponseWindowSize;

            m_rrc->m_rrcSapUser->SendRrcConnectionReconfiguration(m_rnti, handoverCommand);

            std::cout << "Switching to HANDOVER_LEAVING state." << std::endl;
            SwitchToState(HANDOVER_LEAVING);
            m_handoverLeavingTimeout = Simulator::Schedule(m_rrc->m_handoverLeavingTimeoutDuration,
                                                           &NrGnbRrc::HandoverLeavingTimeout,
                                                           m_rrc,
                                                           m_rnti);
            std::cout << "Handover started. IMSI: " << m_imsi
                      << ", Source Cell: " << sourceComponentCarrier->GetCellId()
                      << ", Target Cell: " << handoverCommand.mobilityControlInfo.targetPhysCellId
                      << std::endl;
        }
        else
        {
            // Inter-gNB aka X2 handover
            std::cout << "Inter-gNB handover (i.e., X2) for CellId: " << cellId << std::endl;

            NrEpcX2SapProvider::HandoverRequestParams params;
            params.oldGnbUeX2apId = m_rnti;
            params.cause = NrEpcX2SapProvider::HandoverDesirableForRadioReason;
            params.sourceCellId = m_rrc->ComponentCarrierToCellId(m_componentCarrierId);
            params.targetCellId = cellId;
            params.mmeUeS1apId = m_imsi;
            params.ueAggregateMaxBitRateDownlink = 200 * 1000;
            params.ueAggregateMaxBitRateUplink = 100 * 1000;
            params.bearers = GetErabList();

            NrRrcSap::HandoverPreparationInfo hpi;
            hpi.asConfig.sourceUeIdentity = m_rnti;
            hpi.asConfig.sourceDlCarrierFreq = sourceComponentCarrier->GetDlEarfcn();
            hpi.asConfig.sourceMeasConfig = m_rrc->m_ueMeasConfig;
            hpi.asConfig.sourceRadioResourceConfig =
                GetRadioResourceConfigForHandoverPreparationInfo();
            hpi.asConfig.sourceMasterInformationBlock.dlBandwidth =
                sourceComponentCarrier->GetDlBandwidth();
            hpi.asConfig.sourceMasterInformationBlock.systemFrameNumber = 0;

            // ... (continuing as before with `std::cout` for logs)
             hpi.asConfig.sourceSystemInformationBlockType1.cellAccessRelatedInfo.plmnIdentityInfo
                .plmnIdentity = m_rrc->m_sib1.at(m_componentCarrierId)
                                    .cellAccessRelatedInfo.plmnIdentityInfo.plmnIdentity;
            hpi.asConfig.sourceSystemInformationBlockType1.cellAccessRelatedInfo.cellIdentity =
                m_rrc->ComponentCarrierToCellId(m_componentCarrierId);
            hpi.asConfig.sourceSystemInformationBlockType1.cellAccessRelatedInfo.csgIndication =
                m_rrc->m_sib1.at(m_componentCarrierId).cellAccessRelatedInfo.csgIndication;
            hpi.asConfig.sourceSystemInformationBlockType1.cellAccessRelatedInfo.csgIdentity =
                m_rrc->m_sib1.at(m_componentCarrierId).cellAccessRelatedInfo.csgIdentity;
            NrGnbCmacSapProvider::RachConfig rc =
                m_rrc->m_cmacSapProvider.at(m_componentCarrierId)->GetRachConfig();
            hpi.asConfig.sourceSystemInformationBlockType2.radioResourceConfigCommon
                .rachConfigCommon.preambleInfo.numberOfRaPreambles = rc.numberOfRaPreambles;
            hpi.asConfig.sourceSystemInformationBlockType2.radioResourceConfigCommon
                .rachConfigCommon.raSupervisionInfo.preambleTransMax = rc.preambleTransMax;
            hpi.asConfig.sourceSystemInformationBlockType2.radioResourceConfigCommon
                .rachConfigCommon.raSupervisionInfo.raResponseWindowSize = rc.raResponseWindowSize;
            hpi.asConfig.sourceSystemInformationBlockType2.radioResourceConfigCommon
                .rachConfigCommon.txFailParam.connEstFailCount = rc.connEstFailCount;
            hpi.asConfig.sourceSystemInformationBlockType2.freqInfo.ulCarrierFreq =
                sourceComponentCarrier->GetUlEarfcn();
            hpi.asConfig.sourceSystemInformationBlockType2.freqInfo.ulBandwidth =
                sourceComponentCarrier->GetUlBandwidth();

            params.rrcContext = m_rrc->m_rrcSapUser->EncodeHandoverPreparationInformation(hpi);

            std::cout << "Sending Handover Request: "
                      << "Source Cell: " << params.sourceCellId
                      << ", Target Cell: " << params.targetCellId
                      << std::endl;
            if (!m_rrc->m_x2SapProvider)
            {
                std::cerr << "Error: m_x2SapProvider is null. Cannot send handover request." << std::endl;
                return;
            }
            m_rrc->m_x2SapProvider->SendHandoverRequest(params);

            //print 
            std::cout << "Switching to HANDOVER_PREPARATION state." << std::endl;
            SwitchToState(HANDOVER_PREPARATION);
        }
    }
    break;

    default:
        std::cerr << "Error: Method unexpected in state " << std::endl;
        break;
    }
}

void HandoverDecision(Ptr<NrHelper> nrHelper, Ptr<NrPointToPointEpcHelper> epcHelper, NodeContainer& ueNodes, NetDeviceContainer& gnbDevs, 
                     NetDeviceContainer& ueDevs, double pathLossExponent) {
    NS_ASSERT(nrHelper != nullptr);
    NS_ASSERT(epcHelper != nullptr);
    
    if (gnbDevs.GetN() == 0 || ueDevs.GetN() == 0) {
        std::cout << "[WARNING] Empty device containers, skipping handover decision" << std::endl;
        return;
    }

    const double NEIGHBOR_DISTANCE_THRESHOLD = 500.0;
    std::map<uint32_t, uint16_t> currentAttachments;

    // Process each UE
    for (uint32_t i = 0; i < ueDevs.GetN(); ++i) {
        std::cout << "\n[DEBUG] Processing UE " << i << std::endl;
        
        Ptr<NrUeNetDevice> ueDevice = DynamicCast<NrUeNetDevice>(ueDevs.Get(i));
        if (!ueDevice) {
            std::cout << "[WARNING] Invalid UE device at index " << i << std::endl;
            continue;
        }

        // Get current cell ID safely
        uint16_t currentCellId = 0;
        bool hasValidCellId = false;
        
        try {
            currentCellId = ueDevice->GetCellId();
            if (currentCellId > 0) {
                hasValidCellId = true;
                currentAttachments[i] = currentCellId;
                std::cout << "[DEBUG] UE " << i << " current cell ID: " << currentCellId << std::endl;
            } else {
                std::cout << "[WARNING] UE " << i << " has invalid cell ID: " << currentCellId << std::endl;
                continue;
            }
        } catch (const std::exception& e) {
            std::cout << "[ERROR] Error getting cell ID for UE " << i << ": " << e.what() << std::endl;
            continue;
        }

        if (!hasValidCellId) {
            std::cout << "[WARNING] UE " << i << " is not attached to any cell" << std::endl;
            continue;
        }

        // Verify UE's RRC layer before proceeding
        Ptr<NrUeRrc> ueRrc = ueDevice->GetRrc();
        if (!ueRrc) {
            std::cout << "[ERROR] UE " << i << " has no RRC layer" << std::endl;
            continue;
        }

        uint16_t rnti = ueRrc->GetRnti();
        std::cout << "[DEBUG] UE " << i << " RNTI: " << rnti << std::endl;
        if (rnti == 0) {
            std::cout << "[ERROR] UE " << i << " has invalid RNTI" << std::endl;
            continue;
        }
        
        // Find current serving gNB
        Ptr<NrGnbNetDevice> servingGnb = nullptr;
        double servingRsrp = -std::numeric_limits<double>::infinity();
        
        std::cout << "[DEBUG] Searching for serving gNB..." << std::endl;
        for (uint32_t j = 0; j < gnbDevs.GetN(); ++j) {
            Ptr<NrGnbNetDevice> gnbDevice = DynamicCast<NrGnbNetDevice>(gnbDevs.Get(j));
            if (!gnbDevice || !gnbDevice->GetNode()) {
                continue;
            }

            if (gnbDevice->GetCellId() == currentCellId) {
                servingGnb = gnbDevice;
                
                // Verify RRC and PHY layers exist
                Ptr<NrGnbRrc> gnbRrc = gnbDevice->GetRrc();
                Ptr<NrGnbPhy> gnbPhy = gnbDevice->GetPhy(0);
                
                if (!gnbRrc || !gnbPhy) {
                    std::cout << "[ERROR] Serving gNB missing RRC or PHY layer" << std::endl;
                    continue;
                }
                
                Ptr<MobilityModel> gnbMobility = gnbDevice->GetNode()->GetObject<MobilityModel>();
                if (gnbMobility) {
                    double txPowerDbm = gnbPhy->GetTxPower();
                    servingRsrp = CalculateRsrp(ueDevice->GetNode(), gnbDevice->GetNode(), 
                                              txPowerDbm, pathLossExponent);
                    std::cout << "[DEBUG] Found serving gNB (CellId: " << currentCellId 
                              << ") with RSRP: " << servingRsrp << " dB" << std::endl;
                }
                break;
            }
        }

        if (!servingGnb) {
            std::cout << "[WARNING] No serving gNB found for UE " << i << std::endl;
            continue;
        }

        // Find best neighbor
        double maxRsrp = servingRsrp;
        Ptr<NrGnbNetDevice> targetGnbDevice = nullptr;
        uint16_t targetCellId = 0;

        Ptr<MobilityModel> ueMobility = ueDevice->GetNode()->GetObject<MobilityModel>();
        if (!ueMobility) {
            std::cout << "[WARNING] UE has no mobility model at index " << i << std::endl;
            continue;
        }

        Vector uePosition = ueMobility->GetPosition();
        std::cout << "[DEBUG] Checking neighboring gNBs..." << std::endl;
        
        // Check neighboring gNBs
        for (uint32_t j = 0; j < gnbDevs.GetN(); ++j) {
            Ptr<NrGnbNetDevice> gnbDevice = DynamicCast<NrGnbNetDevice>(gnbDevs.Get(j));
            if (!gnbDevice || !gnbDevice->GetNode() || gnbDevice->GetCellId() == currentCellId) {
                continue;
            }

            // Verify RRC and PHY layers exist for neighbor
            Ptr<NrGnbRrc> gnbRrc = gnbDevice->GetRrc();
            Ptr<NrGnbPhy> gnbPhy = gnbDevice->GetPhy(0);
            
            if (!gnbRrc || !gnbPhy) {
                std::cout << "[WARNING] Neighboring gNB missing RRC or PHY layer" << std::endl;
                continue;
            }

            Ptr<MobilityModel> gnbMobility = gnbDevice->GetNode()->GetObject<MobilityModel>();
            if (!gnbMobility) {
                continue;
            }

            Vector gnbPosition = gnbMobility->GetPosition();
            double distance = CalculateDistance(uePosition, gnbPosition);

            if (distance <= NEIGHBOR_DISTANCE_THRESHOLD) {
                double txPowerDbm = gnbPhy->GetTxPower();
                double rsrp = CalculateRsrp(ueDevice->GetNode(), gnbDevice->GetNode(), 
                                          txPowerDbm, pathLossExponent);

                std::cout << "[DEBUG] Neighbor gNB (CellId: " << gnbDevice->GetCellId() 
                          << ") RSRP: " << rsrp << " dB" << std::endl;

                if (rsrp > maxRsrp) {
                    maxRsrp = rsrp;
                    targetGnbDevice = gnbDevice;
                    targetCellId = gnbDevice->GetCellId();
                }
            }
        }

        // Perform handover if better neighbor found
        if (targetGnbDevice && targetCellId != currentCellId && maxRsrp > servingRsrp) {
            std::cout << "\n[INFO] Triggering handover for UE " << i 
                      << "\n  From CellId: " << currentCellId
                      << " (RSRP: " << servingRsrp << " dB)"
                      << "\n  To CellId: " << targetCellId 
                      << " (RSRP: " << maxRsrp << " dB)"
                      << "\n  RSRP improvement: " << maxRsrp - servingRsrp << " dB" << std::endl;

            try {
                // Additional verification of serving gNB's RRC
                Ptr<NrGnbRrc> servingRrc = servingGnb->GetRrc();
                if (!servingRrc) {
                    std::cout << "[ERROR] Serving gNB has no RRC" << std::endl;
                    continue;
                }

                // Verify target gNB's RRC
                Ptr<NrGnbRrc> targetRrc = targetGnbDevice->GetRrc();
                if (!targetRrc) {
                    std::cout << "[ERROR] Target gNB has no RRC" << std::endl;
                    continue;
                }

                // Send handover request with additional checks
                std::cout << "[DEBUG] Starting handover process..." << std::endl;
                std::cout << "[DEBUG] UE RNTI: " << rnti << std::endl;
                std::cout << "[DEBUG] Target Cell ID: " << targetCellId << std::endl;

                // servingRrc->SendHandoverRequest(rnti, targetCellId);
                //ManualHandover(nrHelper, epcHelper, ueDevs, gnbDevs, i, targetCellId, servingGnb);
                //print the UeManager of serving RRc
                Ptr<NrUeManager> ueManager = servingRrc->GetUeManager(rnti);
                if (!ueManager) {
                    std::cout << "[ERROR] UE Manager not found for serving gNB" << std::endl;
                    continue;
                }
                if (ueRrc->GetState() != NrUeRrc::CONNECTED_NORMALLY) {
                    std::cout << "[WARNING] UE " << i 
                            << " is not in CONNECTED_NORMALLY state (current state: " 
                            << ueRrc->GetState() << "). Skipping handover." << std::endl;
                    continue;
                }
                nrHelper->HandoverRequest(Seconds(0.1), ueDevice, servingGnb, targetCellId);
                //ueManager->PrepareHandover(targetCellId);
                
                std::cout << "[DEBUG] Handover request sent successfully" << std::endl;
            }
            catch (const std::exception& e) {
                std::cout << "[ERROR] Error during handover for UE " << i << ": " << e.what() << std::endl;
            }
        }
    }

    // Schedule the next handover decision
    // const double HANDOVER_CHECK_INTERVAL = 1.0; // Check every 1s
    // Simulator::Schedule(Seconds(HANDOVER_CHECK_INTERVAL), &HandoverDecision, 
    //                    nrHelper, epcHelper, ueNodes, gnbDevs, ueDevs, pathLossExponent);
    
    // std::cout << "\n[INFO] Handover decision completed. Next check scheduled in " 
    //           << HANDOVER_CHECK_INTERVAL << " seconds.\n" << std::endl;
}


// Helper function to calculate distance between two points
double CalculateDistance(const Vector3D& pos1, const Vector3D& pos2) {
    double dx = pos1.x - pos2.x;
    double dy = pos1.y - pos2.y;
    double dz = pos1.z - pos2.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// Custom event to ensure setup is complete before handover
void WaitForSetupAndTriggerHandover(Ptr<NrHelper> nrHelper, Ptr<NrPointToPointEpcHelper> epcHelper, NodeContainer& ueNodes, 
                                    NetDeviceContainer& gnbDevs, NetDeviceContainer& ueDevs, double pathLossExponent) {
    // Check that setup is complete before scheduling handover decision
    bool setupComplete = true;

    // Ensure all UE devices are attached and have proper RRC connections
    for (uint32_t i = 0; i < ueDevs.GetN(); ++i) {
        Ptr<NrUeNetDevice> ueDevice = DynamicCast<NrUeNetDevice>(ueDevs.Get(i));
        if (!ueDevice || !ueDevice->GetRrc()) {
            setupComplete = false;
            break;
        }
    }

    // If setup is complete, trigger handover decision
    if (setupComplete) {
        // Now that setup is complete, trigger handover decision
        std::cout << "Setup complete. Triggering handover decision..." << std::endl;
        Simulator::Schedule(Seconds(0.0), &HandoverDecision, nrHelper, epcHelper, ueNodes, gnbDevs, ueDevs, pathLossExponent);
    }
    else {
        // If setup isn't complete, keep checking after some time
        std::cout << "Setup not complete. Waiting for UE setup..." << std::endl;
        Simulator::Schedule(Seconds(0.5), &WaitForSetupAndTriggerHandover, nrHelper, epcHelper, ueNodes, gnbDevs, ueDevs, pathLossExponent);
    }
}

// Add this as a global variable or class member to store previous measurements
std::map<FlowId, uint64_t> lastTotalBytes;

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

void PrintUeConnectionStatus(NetDeviceContainer ueDevs, NetDeviceContainer gnbDevs, Ipv4InterfaceContainer ueIpIface) {
    std::cout << "\n=== UE Connection Status ===" << std::endl;
    
    for (uint32_t i = 0; i < ueDevs.GetN(); ++i) {
        Ptr<NrUeNetDevice> ueDevice = DynamicCast<NrUeNetDevice>(ueDevs.Get(i));
        uint16_t cellId = ueDevice->GetCellId();
        
        // Get UE position
        Ptr<MobilityModel> ueMobility = ueDevice->GetNode()->GetObject<MobilityModel>();
        Vector uePos = ueMobility->GetPosition();
        
        // Find connected gNB
        Ptr<NrGnbNetDevice> connectedGnb = nullptr;
        Vector gnbPos;
        for (uint32_t j = 0; j < gnbDevs.GetN(); ++j) {
            Ptr<NrGnbNetDevice> gnbDevice = DynamicCast<NrGnbNetDevice>(gnbDevs.Get(j));
            if (gnbDevice->GetCellId() == cellId) {
                connectedGnb = gnbDevice;
                Ptr<MobilityModel> gnbMobility = gnbDevice->GetNode()->GetObject<MobilityModel>();
                gnbPos = gnbMobility->GetPosition();
                break;
            }
        }
        
        double distance = -1;
        if (connectedGnb) {
            distance = sqrt(pow(uePos.x - gnbPos.x, 2) + pow(uePos.y - gnbPos.y, 2));
        }
        
        std::cout << "UE " << i << ":" << std::endl;
        std::cout << "  IP Address: " << ueIpIface.GetAddress(i) << std::endl;
        std::cout << "  Position: (" << uePos.x << ", " << uePos.y << ")" << std::endl;
        std::cout << "  Connected to gNB: " << (connectedGnb ? "Yes" : "No") << std::endl;
        if (connectedGnb) {
            std::cout << "  Connected gNB ID: " << cellId << std::endl;
            std::cout << "  Distance to gNB: " << distance << " meters" << std::endl;
        }
        //print rsrp using CalculateRSRP function
        double rsrp = CalculateRsrp(ueDevice->GetNode(), connectedGnb->GetNode(), connectedGnb->GetPhy(0)->GetTxPower(), 3.5);
        std::cout << "  RSRP: " << rsrp << " dBm" << std::endl;
        std::cout << std::endl;
    }

    // periodic check
    Simulator::Schedule(Seconds(1.0), &PrintUeConnectionStatus, ueDevs, gnbDevs, ueIpIface);
}

void checkUEPosition(NodeContainer& ueNodes){
    for (uint32_t i = 0; i < ueNodes.GetN(); ++i) {
        Ptr<ConstantVelocityMobilityModel> mobility = 
            ueNodes.Get(i)->GetObject<ConstantVelocityMobilityModel>();
        std::cout << "UE " << i << " position: " << mobility->GetPosition() 
                << ", velocity: " << mobility->GetVelocity() << std::endl;
    }
}

int main(int argc, char *argv[]) {
    double simTime = 60;
    int numGnb = 10;
    int numUes = 10;
    double minPowerDbm = 10.0; //10
    double maxPowerDbm = 40.0;
    double minSpeed = 1.0;
    double maxSpeed = 3.0;
    double pathLossExponent = 3.5; //3.5
    double frequency = 28e9;
    double bandwidth = 100e6;     // 100 MHz bandwidth
    double maxLength = 500.0; // Maximum length of the simulation area, if UE and GNB too far throughput may be 0
    BandwidthPartInfo::Scenario scenario = BandwidthPartInfo::UMa; // Urban Macro scenario

    // Enable logging for the animation interface
    // LogComponentEnable("AnimationInterface", LOG_LEVEL_INFO);

    std::cout << "Starting 5G small cell simulation with " << numGnb 
              << " gNBs and " << numUes << " UEs." << std::endl;

    NodeContainer gnbNodes;
    gnbNodes.Create(numGnb);
    NodeContainer ueNodes;
    ueNodes.Create(numUes);

    RandomizeCioValues(numGnb, minCio, maxCio);

    Ptr<NrPointToPointEpcHelper> epcHelper = CreateObject<NrPointToPointEpcHelper>(); // EPC helper
    Ptr<IdealBeamformingHelper> idealBeamformingHelper = CreateObject<IdealBeamformingHelper>(); // Beamforming helper
    Ptr<NrHelper> nrHelper = CreateObject<NrHelper>(); // Main NR helper
    nrHelper->SetBeamformingHelper(idealBeamformingHelper); // Set the beamforming helper
    nrHelper->SetEpcHelper(epcHelper); // Set the EPC helper

    MobilityHelper gnbMobility;
    gnbMobility.SetPositionAllocator("ns3::RandomBoxPositionAllocator",
                                "X", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=" + std::to_string(maxLength) + "]"),
                                "Y", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=" + std::to_string(maxLength) + "]"),
                                "Z", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
    gnbMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    gnbMobility.Install(gnbNodes);

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
        double direction = (2 * M_PI) * ((double) rand() / RAND_MAX);  // Random direction

        // Set velocity with random speed and direction
        mobility->SetVelocity(Vector(speed * cos(direction), speed * sin(direction), 0));
        
        // Debug output for mobility configuration
        std::cout << "Set UE " << i << " speed to " << speed << " m/s, direction to " << direction << " radians." << std::endl;
    }

    // Optionally, you can check positions during simulation:
    checkUEPosition(ueNodes);

    // NR configuration for communication
    BandwidthPartInfoPtrVector allBwps;
    CcBwpCreator ccBwpCreator; // Bandwidth part creator
    const uint8_t numCcPerBand = 1;

    // Define the configuration of the operation band (frequency, bandwidth, and scenario)
    CcBwpCreator::SimpleOperationBandConf bandConf(frequency, bandwidth, numCcPerBand, scenario);

    // Create the operation band based on the configuration
    OperationBandInfo band = ccBwpCreator.CreateOperationBandContiguousCc(bandConf);
    nrHelper->InitializeOperationBand(&band); // Initialize the operation band with the helper
    allBwps = CcBwpCreator::GetAllBwps({band}); // Get all bandwidth parts

    // Configure ideal RRC protocol
    nrHelper->SetGnbPhyAttribute("Numerology", UintegerValue(2));
    // Set ideal beamforming method
    idealBeamformingHelper->SetAttribute("BeamformingMethod",
                                         TypeIdValue(DirectPathBeamforming::GetTypeId()));

    // Configure the scheduler with explicit RNTI management
    nrHelper->SetSchedulerTypeId(TypeId::LookupByName("ns3::NrMacSchedulerTdmaPF"));

    // Configure UEs antennas
    nrHelper->SetUeAntennaAttribute("NumRows", UintegerValue(2));
    nrHelper->SetUeAntennaAttribute("NumColumns", UintegerValue(4));
    nrHelper->SetUeAntennaAttribute("AntennaElement",
                                    PointerValue(CreateObject<IsotropicAntennaModel>()));

    // Configure gNBs antennas
    nrHelper->SetGnbAntennaAttribute("NumRows", UintegerValue(8));
    nrHelper->SetGnbAntennaAttribute("NumColumns", UintegerValue(8));
    nrHelper->SetGnbAntennaAttribute("AntennaElement",
                                     PointerValue(CreateObject<IsotropicAntennaModel>()));

    nrHelper->SetHandoverAlgorithmType("ns3::A3RsrpHandoverAlgorithm");
    nrHelper->SetHandoverAlgorithmAttribute("Hysteresis", DoubleValue(0.5));
    nrHelper->SetHandoverAlgorithmAttribute("TimeToTrigger", TimeValue(Seconds(0.1)));

    // Install NR devices (gNB and UE)
    NetDeviceContainer ueDevs = nrHelper->InstallUeDevice(ueNodes, allBwps); // Install on vehicles (UEs)
    NetDeviceContainer gnbDevs = nrHelper->InstallGnbDevice(gnbNodes, allBwps); // Install on base stations (gNBs)

    nrHelper->AddX2Interface(gnbNodes);
    std::cout << "X2 interface added between gNBs" << std::endl;

    // Assign random streams to devices for randomization purposes
    int64_t randomStream = 1;
    randomStream += nrHelper->AssignStreams(gnbDevs, randomStream);
    randomStream += nrHelper->AssignStreams(ueDevs, randomStream);
    
    // Set random transmission power for each gNB
    for (uint32_t i = 0; i < gnbDevs.GetN(); ++i) {
        Ptr<NrGnbNetDevice> gnb = DynamicCast<NrGnbNetDevice>(gnbDevs.Get(i));
        double power = minPowerDbm + (maxPowerDbm - minPowerDbm) * ((double) rand() / RAND_MAX);
        gnb->GetPhy(0)->SetTxPower(power);
    }

    // Update the configuration for gNB and UE devices
    for (auto it = gnbDevs.Begin(); it != gnbDevs.End(); ++it)
    {
        DynamicCast<NrGnbNetDevice>(*it)->UpdateConfig();
    }

    for (auto it = ueDevs.Begin(); it != ueDevs.End(); ++it)
    {
        DynamicCast<NrUeNetDevice>(*it)->UpdateConfig();
    }

    // Create the internet stack and install on the UEs
    Ptr<Node> pgw = epcHelper->GetPgwNode(); // Get the PGW node (packet gateway)
    NodeContainer remoteHostContainer;
    remoteHostContainer.Create(1); // Create the remote host
    Ptr<Node> remoteHost = remoteHostContainer.Get(0);
    InternetStackHelper internet;
    internet.Install(remoteHostContainer); // Install internet stack on the remote host

    // Connect the remote host to the PGW and setup routing
    PointToPointHelper p2ph;
    p2ph.SetDeviceAttribute("DataRate", DataRateValue(DataRate("100Gb/s"))); // Data rate of the point-to-point link
    p2ph.SetDeviceAttribute("Mtu", UintegerValue(2500)); // MTU size
    p2ph.SetChannelAttribute("Delay", TimeValue(Seconds(0.010))); // Link delay
    NetDeviceContainer internetDevices = p2ph.Install(pgw, remoteHost); // Install devices on PGW and remote host

    // Assign IP addresses to the remote host
    Ipv4AddressHelper ipv4h;
    ipv4h.SetBase("1.0.0.0", "255.0.0.0"); // Base IP address
    Ipv4InterfaceContainer internetIpIfaces = ipv4h.Assign(internetDevices);
    Ipv4StaticRoutingHelper ipv4RoutingHelper;

    // Setup routing for the remote host
    Ptr<Ipv4StaticRouting> remoteHostStaticRouting =
        ipv4RoutingHelper.GetStaticRouting(remoteHost->GetObject<Ipv4>());
    remoteHostStaticRouting->AddNetworkRouteTo(Ipv4Address("7.0.0.0"), Ipv4Mask("255.0.0.0"), 1);

    // Configure internet protocol on the UEs and gNBs
    internet.Install(ueNodes);

    // Setup IP addresses for the UEs
    Ipv4InterfaceContainer ueIpIface;
    ueIpIface = epcHelper->AssignUeIpv4Address(NetDeviceContainer(ueDevs));

    // Attach UEs to the network (associate with gNB)
    nrHelper->AttachToClosestGnb(ueDevs, gnbDevs);
    std::cout << "UEs attached to the closest gNB." << std::endl;

    // Wait a bit to ensure attachment is complete
    Simulator::Schedule(Seconds(0.2), [&]() {
        // Now activate the data radio bearer
        NrEpsBearer bearer(NrEpsBearer::NGBR_VIDEO_TCP_DEFAULT);
        for (uint32_t i = 0; i < ueDevs.GetN(); ++i) {
            nrHelper->ActivateDataRadioBearer(ueDevs.Get(i), bearer);
        }
    });

    std::cout << "UEs attached to the closest gNB and bearers being activated." << std::endl;
    PrintUeConnectionStatus(ueDevs, gnbDevs, ueIpIface);

    PrintGnbMeasurements(gnbDevs, ueNodes, pathLossExponent);
    PrintConnectedGnbRsrp(gnbDevs, ueDevs, pathLossExponent, nrHelper);

    // Setup routes on the UEs
    for (uint32_t u = 0; u < ueNodes.GetN(); ++u)
    {
        Ptr<Node> ueNode = ueNodes.Get(u);
        Ptr<Ipv4StaticRouting> ueStaticRouting = ipv4RoutingHelper.GetStaticRouting(ueNode->GetObject<Ipv4>());
        ueStaticRouting->SetDefaultRoute(epcHelper->GetUeDefaultGatewayAddress(), 1);
    }

    uint16_t serverPort = 8080;
    UdpServerHelper server(serverPort);
    ApplicationContainer serverApp = server.Install(ueNodes.Get(0));
        
    // Debug print
    std::cout << "Installing UDP server on UE " << 0 << " (IP: " << ueIpIface.GetAddress(0) 
            << ") port " << serverPort << std::endl;
    serverApp.Start(Seconds(0.0));
    serverApp.Stop(Seconds(simTime));

    // Then create UDP clients
    ApplicationContainer clientApps;
    // Install a client on every UE sending to UE0
    for (uint32_t i = 1; i < ueNodes.GetN(); ++i) {
        // All clients send to UE0
        UdpClientHelper client(ueIpIface.GetAddress(0), serverPort);
        client.SetAttribute("MaxPackets", UintegerValue(0));
        client.SetAttribute("Interval", TimeValue(MilliSeconds(100)));
        client.SetAttribute("PacketSize", UintegerValue(1024));
        
        ApplicationContainer tempApp = client.Install(ueNodes.Get(i));
        clientApps.Add(tempApp);
        
        std::cout << "Installing UDP client on UE " << i 
                << " sending to UE0 (" << ueIpIface.GetAddress(0) << ":" << serverPort << ")" << std::endl;
    }
    clientApps.Start(Seconds(1.0));
    clientApps.Stop(Seconds(simTime));
    

    Simulator::Schedule(Seconds(0.5), &WaitForSetupAndTriggerHandover, nrHelper, epcHelper, ueNodes, gnbDevs, ueDevs, pathLossExponent);
    
    Ptr<FlowMonitor> flowMonitor;
    FlowMonitorHelper flowHelper;
    flowMonitor = flowHelper.InstallAll();

    // After the simulation, calculate and print the throughput
    flowMonitor->CheckForLostPackets();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(flowHelper.GetClassifier());
    // Start the throughput calculation function
    // Schedule the first throughput calculation
    //Simulator::Schedule(Seconds(1.0), &CalculateThroughput, flowMonitor, classifier, ueIpIface);

    // p2ph.EnablePcapAll("udp_debug");
    Simulator::Stop(Seconds(simTime));
    Simulator::Run();
    Simulator::Destroy();

    std::cout << "Simulation finished." << std::endl;
    return 0;
}