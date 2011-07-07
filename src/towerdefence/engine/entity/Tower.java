package towerdefence.engine.entity;

import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.newdawn.slick.Color;
import org.newdawn.slick.GameContainer;
import org.newdawn.slick.Graphics;
import org.newdawn.slick.Image;
import org.newdawn.slick.Input;
import org.newdawn.slick.SlickException;
import org.newdawn.slick.UnicodeFont;
import org.newdawn.slick.font.effects.ColorEffect;
import org.newdawn.slick.geom.Circle;
import org.newdawn.slick.geom.Vector2f;
import org.newdawn.slick.state.StateBasedGame;
import towerdefence.GameplayState;
import towerdefence.engine.Player;
import towerdefence.engine.component.Component;

/**
 *
 * Tower - An entity that deals with distance checking against critters
 * and shoots critters
 * 
 *
 * @author Jiv Dhaliwal <jivdhaliwal@gmail.com>
 */
public class Tower extends Entity {
    
        // Tower types
    public final static int NORMAL = 0;
    public final static int FIRE = 1;
    public final static int ICE = 2;
    public final static int BOSS = 3;

    ArrayList<Critter> critterList = null;
    private Critter targetCritter = null;
    
    private Image[] sprites;

    private float range;
    float damagePerSec;
    private int type;

    private int shootingCounter;

    boolean isShooting;
    private boolean isPlaced=true;
    private boolean isActive;
    
    private Circle circle=null;
    private int mouseXTile;
    private int mouseYTile;
    
    private final UnicodeFont unicodeFont;
    
    String costOverlay;
    String overlay;
    String iceDetails = "150% damage\nto fire types";
    String fireDetails = "150% damage\nto ice types";
    


    public Tower(String id, boolean isActive) throws SlickException{
        super(id);

        this.rotation=0;
        this.isActive = isActive;
        
        unicodeFont = new UnicodeFont("fonts/pf_tempesta_seven.ttf", 8, false, false);
        unicodeFont.getEffects().add(new ColorEffect(java.awt.Color.white));
        
    }

    /*
     * Update the list of critters the tower iterates through to find a target
     */
    public void updateCritterList(ArrayList<Critter> critterList) {
        this.critterList = critterList;
    }

    private void renderTextOverlay(Graphics gr) {
        String tempOverlay;
        
        if(isActive) {
            tempOverlay = "Sale Value : $"+
                    (Player.getInstance().getTowerCost(type))/2+"\n"+overlay;
            if(type==FIRE) {
                tempOverlay+=fireDetails;
            } else if(type==ICE) {
                tempOverlay+=iceDetails;
            }
        } else {
            tempOverlay = costOverlay + overlay;
            if(type==FIRE) {
                tempOverlay+=fireDetails;
            } else if(type==ICE) {
                tempOverlay+=iceDetails;
            }
        }
        
        if (Player.getInstance().getCash() - Player.getInstance().getTowerCost(type) >= 0) {
            unicodeFont.drawString(42 + (21 * 32), 416,tempOverlay);
        } else {
            if(!isPlaced) {
                gr.drawString("Not enough cash", position.x+32, position.y+32);
            }
            unicodeFont.drawString(42 + (21 * 32), 416,costOverlay, Color.red);
            unicodeFont.drawString(42 + (21 * 32), 431,overlay);
        }
    }

    private void findClosestCritter() {

        // If critters on the map
        if(critterList!=null) {

            // If no critters is locked on
            if(getTargetCritter()==null) {
                Critter tempTarget = new Critter("test");
                // Set initial position very far away from the map
                tempTarget.setPosition(new Vector2f(-1000f,-1000f));
                for(Critter enemy : critterList) {
                    float critterDistance = this.getPosition().distance(enemy.getPosition());
                    float tempDistance = this.getPosition().distance(tempTarget.getPosition());

                    // First check the critter is in range of the tower
                    // Then check if it is closer than the tempCritter
                    // Do this for all critters to eventually find the closest one
                    if(critterDistance < range) {
                        if(critterDistance < tempDistance) {
                            setTargetCritter(enemy);
                            tempTarget=enemy;
                        }
                    }
                    // Remnants of the multicolour lazers
                    // colourCounter++;
                }
            } else if (getTargetCritter()!=null) {
                // If the critter is out of range or dead, find a new target
                // Else shoot the
                if(this.getPosition().distance(getTargetCritter().getPosition()) >= range) {
                    setTargetCritter(null);
                } else if(getTargetCritter().isDead()||getTargetCritter().isDelete()) {
                    setTargetCritter(null);
                } else {
                    shootCritter(getTargetCritter());
                }
            }
        }
    }

    private void guiOverlay(Graphics gr) {
        
        costOverlay = "Cost : $" + Player.getInstance().getTowerCost(type)+"\n";
        overlay = "Range = " + (int)range +"\nDPS = " + (int)damagePerSec+"\n";
        
        if (!isPlaced) {
            if (circle != null) {
                gr.draw(circle);
            }
            renderTextOverlay(gr);
        } else if (isPlaced && mouseXTile == getTilePosition().x && mouseYTile == getTilePosition().y) {
            if (isActive) {
                if(!GameplayState.towerSelected) {
                    if (circle != null) {
                        gr.draw(circle);
                    }
                    renderTextOverlay(gr);
                }
            } else {
                renderTextOverlay(gr);
            }
            
            
        }
    }

    private void shootCritter(Critter critter) {
        if(critter.getType()==Critter.FIRE && type==Tower.ICE) {
            critter.takeDamage((damagePerSec*1.5f)/10f);   
        } else if(critter.getType()==Critter.ICE && type==Tower.FIRE) {
            critter.takeDamage((damagePerSec*1.5f)/10f);   
        } else {
            critter.takeDamage(damagePerSec/10f);
        }
        
    }
    
    /**
     * @param sprites the sprites to set
     */
    public void setSprites(Image[] sprites) {
        this.sprites = sprites;
    }
    
    @Override
    public void setType(int type) {
        this.type=type;
        range = GameplayState.towerRange[type];
        damagePerSec = GameplayState.baseDPS[type];
        circle.setRadius(range);
        circle.setCenterX(position.x+(GameplayState.TILESIZE/2));
        circle.setCenterY(position.y+(GameplayState.TILESIZE/2));
    }
    
    @Override
    public void setPosition(Vector2f position)
    {
        this.position = position;
        if(circle==null) {
            circle = new Circle(position.x+(GameplayState.TILESIZE/2),position.y+(GameplayState.TILESIZE/2),range);
        } else {
            circle.setCenterX(position.x+(GameplayState.TILESIZE/2));
            circle.setCenterY(position.y+(GameplayState.TILESIZE/2));
        }
    }

    @Override
    public void update(GameContainer gc, StateBasedGame sb, int delta)
    {
        Input i = gc.getInput();
        mouseXTile = (int) Math.floor((i.getAbsoluteMouseX() / GameplayState.TILESIZE));
        mouseYTile = (int) Math.floor((i.getAbsoluteMouseY() / GameplayState.TILESIZE));
        
        
        if(isActive && isPlaced && mouseXTile == getTilePosition().x && mouseYTile == getTilePosition().y){
            if(i.isKeyPressed(Input.KEY_DELETE) || i.isKeyPressed(Input.KEY_NUMPAD0)) {
                Player.getInstance().sellTower(type);
                this.killEntity();
            }
        }
        
        try {
            unicodeFont.loadGlyphs(100);
        } catch (SlickException ex) {
            Logger.getLogger(Tower.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        shootingCounter-=delta;

        if (isActive) {
            if (shootingCounter <= 0) {
                findClosestCritter();
                shootingCounter = 100;
            }
        }


        for(Component component : components)
        {
            component.update(gc,sb,delta);
        }

    }

    @Override
    public void render(GameContainer gc, StateBasedGame sb, Graphics gr)
    {   
        if(renderComponent != null) {
            renderComponent.render(gc, sb, gr);
        }
        // Laser shooting
        // Check tower has a target
        if (getTargetCritter() != null) {

            gr.rotate(this.getPosition().x + 16, this.getPosition().y + 16,
                    (float) (getTargetCritter().getPosition().sub(this.getPosition())).getTheta()-90);
            
            // Draw lazer and extend it using the distance from the tower to the
            // target critter
            sprites[2].draw(this.getPosition().x, this.getPosition().y+16, 32,
                    this.getPosition().distance(getTargetCritter().getPosition()));
            // Draw tower's turret (which will rotate towards critters
            sprites[1].draw(this.getPosition().x,this.getPosition().y);
            gr.rotate(this.getPosition().x + 16, this.getPosition().y + 16,
                    (float) -(targetCritter.getPosition().sub(this.getPosition())).getTheta()+90);
        }
        
        guiOverlay(gr);
        
    }

    /**
     * @param isActive the isActive to set
     */
    public void setIsActive(boolean isActive) {
        this.isActive = isActive;
    }

    /**
     * @param isPlaced the isPlaced to set
     */
    public void setIsPlaced(boolean isPlaced) {
        this.isPlaced = isPlaced;
    }

    /**
     * @return the type
     */
    @Override
    public int getType() {
        return type;
    }

    /**
     * @return the targetCritter
     */
    public Critter getTargetCritter() {
        return targetCritter;
    }

    /**
     * @param targetCritter the targetCritter to set
     */
    public void setTargetCritter(Critter targetCritter) {
        this.targetCritter = targetCritter;
    }
    
}
